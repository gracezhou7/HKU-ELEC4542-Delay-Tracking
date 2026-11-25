"""Dataset sequence discovery and deterministic sequence-level splits."""

import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Set

from data.annotations import SequenceSpec, _read_seqinfo

DEFAULT_SEED = 4542
SPLITS = ("train", "val", "test")


def split_names(
    names: Iterable[str],
    counts: Mapping[str, int],
    seed: int = DEFAULT_SEED,
) -> Dict[str, List[str]]:
    """Shuffle unique names reproducibly and assign the requested split sizes."""
    ordered_names = sorted(set(names))
    if sum(counts.values()) != len(ordered_names):
        raise ValueError(
            "Split sizes total {}, but {} unique sequences were found".format(
                sum(counts.values()), len(ordered_names)
            )
        )
    if any(count < 0 for count in counts.values()):
        raise ValueError("Split sizes cannot be negative")

    shuffled = list(ordered_names)
    random.Random(seed).shuffle(shuffled)
    result: Dict[str, List[str]] = {}
    offset = 0
    for split in SPLITS:
        count = counts.get(split, 0)
        result[split] = sorted(shuffled[offset : offset + count])
        offset += count
    return result


def _sequence_dirs(parent: Path) -> List[Path]:
    if not parent.is_dir():
        raise FileNotFoundError("Dataset directory does not exist: {}".format(parent))
    return sorted(
        path.parent.parent
        for path in parent.glob("*/gt/gt.txt")
        if path.is_file()
    )


def _spec(
    sequence_dir: Path,
    name: str,
    fps: float,
    annotation_format: str = "mot",
) -> SequenceSpec:
    if annotation_format == "otb":
        preferred = sequence_dir / "groundtruth_rect.txt"
        candidates = [preferred] if preferred.is_file() else sorted(
            sequence_dir.glob("groundtruth_rect*.txt")
        )
        if not candidates:
            raise FileNotFoundError(
                "Missing OTB ground truth in {}".format(sequence_dir)
            )
        return SequenceSpec(
            name=name, annotation_path=candidates[0], fps=fps, format="otb"
        )
    annotation_path = sequence_dir / "gt" / "gt.txt"
    if not annotation_path.is_file():
        raise FileNotFoundError("Missing MOT ground truth: {}".format(annotation_path))
    return SequenceSpec(name=name, annotation_path=annotation_path, fps=fps)


def _require_count(
    dataset: str, split: str, actual: int, expected: int, strict: bool = True
) -> None:
    if actual != expected and strict:
        raise ValueError(
            "{} {}: expected {} sequences, found {}".format(
                dataset, split, expected, actual
            )
        )


def _dtb_sequence_dirs(parent: Path) -> List[Path]:
    dirs: Set[Path] = set()
    for path in parent.rglob("gt/gt.txt"):
        if path.is_file():
            dirs.add(path.parent.parent)
    for path in parent.rglob("groundtruth_rect*.txt"):
        if path.is_file():
            dirs.add(path.parent)
    return sorted(dirs)


def discover_splits(
    root: Path,
    seed: int = DEFAULT_SEED,
    strict: bool = True,
) -> Dict[str, Dict[str, SequenceSpec]]:
    """Discover raw datasets and return train/val/test sequence specs.

    Supported layouts:
      root/DanceTrack/{train,val}/<sequence>/gt/gt.txt
      root/MOT17/train/<sequence>-<detector>/gt/gt.txt
      root/MOT17Labels/train/<sequence>-<detector>/gt/gt.txt
      root/DTB70/<sequence>/{gt/gt.txt or groundtruth_rect.txt}

    With strict=False, datasets that are entirely absent are skipped; datasets
    that are present must still contain the expected number of sequences.
    """
    root = Path(root)
    result: Dict[str, Dict[str, SequenceSpec]] = {}

    dance_root = root / "DanceTrack"
    dance_train_root = dance_root / "train"
    dance_val_root = dance_root / "val"
    if dance_train_root.is_dir() and dance_val_root.is_dir():
        dance_train_dirs = _sequence_dirs(dance_train_root)
        dance_val_dirs = _sequence_dirs(dance_val_root)
        _require_count("DanceTrack", "train", len(dance_train_dirs), 40, strict)
        _require_count("DanceTrack", "val", len(dance_val_dirs), 25, strict)
        if len({path.name for path in dance_train_dirs}) != len(dance_train_dirs):
            raise ValueError("DanceTrack train contains duplicate sequence names")
        dance_val_split = split_names(
            (path.name for path in dance_val_dirs),
            {"train": 0, "val": 10, "test": 15},
            seed,
        )
        dance_train = {
            path.name: _spec(path, path.name, 30.0) for path in dance_train_dirs
        }
        dance_val_by_name = {path.name: path for path in dance_val_dirs}
        dance_specs: Dict[str, SequenceSpec] = dict(dance_train)
        dance_specs.update(
            {
                name: _spec(dance_val_by_name[name], name, 30.0)
                for name in dance_val_by_name
            }
        )
        result["DanceTrack"] = {
            "train": dance_train,
            "val": {name: dance_specs[name] for name in dance_val_split["val"]},
            "test": {name: dance_specs[name] for name in dance_val_split["test"]},
        }
    elif strict:
        raise FileNotFoundError(
            "Missing DanceTrack train/val directories under {}".format(dance_root)
        )

    mot_root = root / "MOT17" / "train"
    if not mot_root.is_dir():
        mot_root = root / "MOT17Labels" / "train"
    if mot_root.is_dir():
        mot_dirs = _sequence_dirs(mot_root)
        variants: Dict[str, List[Path]] = {}
        for path in mot_dirs:
            base_name = re.sub(r"-(FRCNN|DPM|SDP)$", "", path.name)
            variants.setdefault(base_name, []).append(path)
        _require_count("MOT17", "unique train", len(variants), 7, strict)
        mot_split = split_names(
            variants.keys(), {"train": 5, "val": 1, "test": 1}, seed
        )
        mot_specs: Dict[str, SequenceSpec] = {}
        for name, candidates in variants.items():
            selected = sorted(
                candidates,
                key=lambda path: (not path.name.endswith("-FRCNN"), path.name),
            )[0]
            fps, _, _ = _read_seqinfo(selected)
            if fps is None:
                raise ValueError(
                    "MOT17 frameRate is missing from {}".format(
                        selected / "seqinfo.ini"
                    )
                )
            mot_specs[name] = _spec(selected, name, fps)
        result["MOT17"] = {
            split: {name: mot_specs[name] for name in mot_split[split]}
            for split in SPLITS
        }
    elif strict:
        raise FileNotFoundError(
            "Missing MOT17 train annotations under {} or {}".format(
                root / "MOT17" / "train", root / "MOT17Labels" / "train"
            )
        )

    dtb_root = root / "DTB70"
    if dtb_root.is_dir():
        dtb_dirs = _dtb_sequence_dirs(dtb_root)
        _require_count("DTB70", "all", len(dtb_dirs), 70, strict)
        dtb_split = split_names(
            (path.name for path in dtb_dirs),
            {"train": 50, "val": 10, "test": 10},
            seed,
        )
        dtb_by_name = {path.name: path for path in dtb_dirs}
        dtb_specs: Dict[str, SequenceSpec] = {}
        for name, sequence_dir in dtb_by_name.items():
            annotation_format = (
                "otb" if list(sequence_dir.glob("groundtruth_rect*.txt")) else "mot"
            )
            dtb_specs[name] = _spec(sequence_dir, name, 30.0, annotation_format)
        result["DTB70"] = {
            split: {name: dtb_specs[name] for name in dtb_split[split]}
            for split in SPLITS
        }
    elif strict:
        raise FileNotFoundError("Missing DTB70 directory under {}".format(dtb_root))

    return result

