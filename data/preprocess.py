"""Validate raw annotations and summarize delayed-motion training windows."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from data.annotations import load_sequence
from data.splits import DEFAULT_SEED, discover_splits
from data.windows import build_anchors, delay_to_future_frames


def summarize(
    root: Path,
    history: int = 8,
    stride: int = 8,
    max_delay_ms: float = 200.0,
    seed: int = DEFAULT_SEED,
    strict: bool = True,
) -> Dict[str, Any]:
    """Load every sequence and report annotation/window statistics."""
    split_specs = discover_splits(root, seed=seed, strict=strict)
    summary: Dict[str, Any] = {
        "root": str(root),
        "config": {
            "history": history,
            "stride": stride,
            "max_delay_ms": max_delay_ms,
            "seed": seed,
            "strict": strict,
        },
        "datasets": {},
        "totals": {
            "sequences": 0,
            "tracks": 0,
            "frames": 0,
            "boxes": 0,
            "windows": 0,
            "empty_sequences": 0,
        },    }

    for dataset, splits in split_specs.items():
        dataset_summary: Dict[str, Any] = {"splits": {}}
        for split, specs in splits.items():
            sequences = []
            split_totals = {
                "sequences": 0,
                "tracks": 0,
                "frames": 0,
                "boxes": 0,
                "windows": 0,
            }
            for name, spec in sorted(specs.items()):
                sequence = load_sequence(spec)
                tracks: Dict[int, Dict[int, Any]] = {}
                for frame_id, boxes in sequence.frames.items():
                    for track_id, box in boxes.items():
                        tracks.setdefault(track_id, {})[frame_id] = box
                max_future_frames = delay_to_future_frames(max_delay_ms, sequence.fps)
                anchors = build_anchors(
                    tracks,
                    history=history,
                    stride=stride,
                    max_future_frames=max_future_frames,
                )
                entry = {
                    "name": name,
                    "format": spec.format,
                    "fps": sequence.fps,
                    "frames": len(sequence.frames),
                    "tracks": len(tracks),
                    "boxes": sum(len(boxes) for boxes in sequence.frames.values()),
                    "max_future_frames": max_future_frames,
                    "windows": len(anchors),
                }
                sequences.append(entry)
                split_totals["sequences"] += 1
                split_totals["tracks"] += entry["tracks"]
                split_totals["frames"] += entry["frames"]
                split_totals["boxes"] += entry["boxes"]
                split_totals["windows"] += entry["windows"]
                if entry["windows"] == 0:
                    summary["totals"]["empty_sequences"] += 1
            dataset_summary["splits"][split] = {
                "totals": split_totals,
                "sequences": sequences,
            }
            for key in ("sequences", "tracks", "frames", "boxes", "windows"):
                summary["totals"][key] += split_totals[key]
        summary["datasets"][dataset] = dataset_summary
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate MOT/OTB annotations and summarize delayed-motion windows."
    )
    parser.add_argument("--root", default="data/raw", help="Raw dataset root")
    parser.add_argument(
        "--output",
        default="data/processed/preprocess_summary.json",
        help="JSON summary path; pass an empty string to skip writing",
    )
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--max-delay-ms", type=float, default=200.0)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip datasets that are entirely absent (useful during development)",
    )
    args = parser.parse_args()

    summary = summarize(
        root=Path(args.root),
        history=args.history,
        stride=args.stride,
        max_delay_ms=args.max_delay_ms,
        seed=args.seed,
        strict=not args.allow_missing,
    )
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
