"""Read MOTChallenge-style ground-truth annotations."""

import configparser
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

Box = Tuple[float, float, float, float]


@dataclass(frozen=True)
class SequenceSpec:
    name: str
    annotation_path: Path
    fps: float
    format: str = "mot"


@dataclass
class SequenceAnnotations:
    name: str
    fps: float
    # Frame number -> track ID -> (left, top, width, height), in pixels.
    frames: Dict[int, Dict[int, Box]]
    image_width: Optional[int] = None
    image_height: Optional[int] = None


def _read_seqinfo(sequence_dir: Path) -> Tuple[Optional[float], Optional[int], Optional[int]]:
    path = sequence_dir / "seqinfo.ini"
    if not path.is_file():
        return None, None, None

    parser = configparser.ConfigParser()
    parser.read(path)
    if not parser.has_section("Sequence"):
        raise ValueError("{} is missing the [Sequence] section".format(path))

    section = parser["Sequence"]
    fps = float(section["frameRate"]) if section.get("frameRate") else None
    width = int(section["imWidth"]) if section.get("imWidth") else None
    height = int(section["imHeight"]) if section.get("imHeight") else None
    return fps, width, height


def _load_otb_sequence(spec: SequenceSpec) -> SequenceAnnotations:
    """Load a single-object OTB box file such as DTB70's groundtruth_rect.txt."""
    sequence_dir = spec.annotation_path.parent
    _, image_width, image_height = _read_seqinfo(sequence_dir)
    if not math.isfinite(spec.fps) or spec.fps <= 0:
        raise ValueError("Invalid frame rate for {}: {}".format(spec.name, spec.fps))

    frames: Dict[int, Dict[int, Box]] = {}
    frame_id = 0
    with spec.annotation_path.open("r", encoding="utf-8") as annotation_file:
        for line_number, raw_line in enumerate(annotation_file, 1):
            line = raw_line.strip()
            if line.startswith("#"):
                continue
            frame_id += 1
            if not line:
                continue
            fields = [
                field for field in line.replace("\t", ",").replace(" ", ",").split(",")
                if field
            ]
            if len(fields) < 4:
                raise ValueError(
                    "{}:{}: expected 4 OTB coordinates".format(
                        spec.annotation_path, line_number
                    )
                )
            try:
                x, y, width, height = (float(value) for value in fields[:4])
            except ValueError as error:
                raise ValueError(
                    "{}:{}: invalid OTB coordinate".format(
                        spec.annotation_path, line_number
                    )
                ) from error
            values = (x, y, width, height)
            if not all(math.isfinite(value) for value in values):
                continue
            if width <= 0 or height <= 0:
                continue
            if x + width <= 0 or y + height <= 0:
                continue
            if image_width is not None and x >= image_width:
                continue
            if image_height is not None and y >= image_height:
                continue
            frames.setdefault(frame_id, {})[1] = (x, y, width, height)

    if not frames:
        raise ValueError("No valid boxes found in {}".format(spec.annotation_path))
    return SequenceAnnotations(
        name=spec.name,
        fps=spec.fps,
        frames=frames,
        image_width=image_width,
        image_height=image_height,
    )


def load_sequence(
    spec: SequenceSpec,
    allowed_class_ids: Optional[Tuple[int, ...]] = (1,),
) -> SequenceAnnotations:
    """Load a sequence from MOT or OTB annotations and filter invalid boxes."""
    if spec.format == "otb" or spec.annotation_path.name.startswith("groundtruth_rect"):
        return _load_otb_sequence(spec)

    sequence_dir = spec.annotation_path.parent.parent
    _, image_width, image_height = _read_seqinfo(sequence_dir)
    if not math.isfinite(spec.fps) or spec.fps <= 0:
        raise ValueError("Invalid frame rate for {}: {}".format(spec.name, spec.fps))

    frames: Dict[int, Dict[int, Box]] = {}
    with spec.annotation_path.open("r", encoding="utf-8") as annotation_file:
        for line_number, raw_line in enumerate(annotation_file, 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = [field.strip() for field in line.split(",")]
            if len(fields) < 6:
                raise ValueError(
                    "{}:{}: expected at least 6 comma-separated MOT fields".format(
                        spec.annotation_path, line_number
                    )
                )
            try:
                frame_id = int(fields[0])
                track_id = int(fields[1])
                x, y, width, height = (float(value) for value in fields[2:6])
                confidence = float(fields[6]) if len(fields) >= 7 else 1.0
                class_id = int(float(fields[7])) if len(fields) >= 8 else 1
                visibility = float(fields[8]) if len(fields) >= 9 else 1.0
            except ValueError as error:
                raise ValueError(
                    "{}:{}: invalid numeric MOT field".format(spec.annotation_path, line_number)
                ) from error

            values = (x, y, width, height, confidence, visibility)
            if not all(math.isfinite(value) for value in values):
                continue
            if frame_id < 1 or track_id < 0 or width <= 0 or height <= 0:
                continue
            if confidence <= 0 or visibility <= 0:
                continue
            if allowed_class_ids is not None and class_id not in allowed_class_ids:
                continue
            if x + width <= 0 or y + height <= 0:
                continue
            if image_width is not None and x >= image_width:
                continue
            if image_height is not None and y >= image_height:
                continue

            frames.setdefault(frame_id, {})[track_id] = (x, y, width, height)

    if not frames:
        raise ValueError("No valid pedestrian boxes found in {}".format(spec.annotation_path))
    return SequenceAnnotations(
        name=spec.name,
        fps=spec.fps,
        frames=frames,
        image_width=image_width,
        image_height=image_height,
    )
