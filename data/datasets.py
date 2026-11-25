"""PyTorch datasets and DataLoader construction for delayed motion windows."""

from typing import Dict, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from data.annotations import SequenceAnnotations, SequenceSpec, load_sequence
from data.features import (
    fourier_delay_encoding,
    normalized_motion,
    normalized_speed,
    relative_motion,
    time_interval,
)
from data.splits import DEFAULT_SEED, discover_splits
from data.windows import box_at_time, build_anchors, delay_to_future_frames


class SequenceWindowDataset(Dataset):
    """Generate per-track history windows and a target box after a simulated delay.

    Each sample contains pixel-space history boxes plus the normalized motion
    features used by the LSTM/Transformer models. The target box is interpolated
    at the exact requested delay, so delays are not quantized to frame boundaries.
    """

    def __init__(
        self,
        sequence: SequenceAnnotations,
        history: int = 8,
        stride: int = 8,
        delay_range_ms: Tuple[float, float] = (33.0, 200.0),
        delay_ms: Optional[float] = None,
        max_delay_ms: Optional[float] = None,
        num_frequencies: int = 8,
    ) -> None:
        if history < 1:
            raise ValueError("history must be at least 1")
        if stride < 1:
            raise ValueError("stride must be at least 1")
        if num_frequencies < 1:
            raise ValueError("num_frequencies must be at least 1")
        low, high = delay_range_ms
        if low <= 0 or high < low:
            raise ValueError("delay_range_ms must satisfy 0 < min <= max")
        if delay_ms is not None and delay_ms <= 0:
            raise ValueError("delay_ms must be positive")

        if max_delay_ms is None:
            max_delay_ms = max(high, delay_ms if delay_ms is not None else high)
        if max_delay_ms <= 0:
            raise ValueError("max_delay_ms must be positive")
        if delay_ms is not None and delay_ms > max_delay_ms:
            raise ValueError("delay_ms cannot exceed max_delay_ms")

        self.sequence = sequence
        self.history = history
        self.stride = stride
        self.delay_range_ms = delay_range_ms
        self.fixed_delay_ms = delay_ms
        self.max_delay_ms = max_delay_ms
        self.num_frequencies = num_frequencies
        self.max_future_frames = delay_to_future_frames(max_delay_ms, sequence.fps)
        self._tracks: Dict[int, Dict[int, Tuple[float, float, float, float]]] = {}
        for frame_id, boxes in sequence.frames.items():
            for track_id, box in boxes.items():
                self._tracks.setdefault(track_id, {})[frame_id] = box

        self._anchors = build_anchors(
            self._tracks,
            history=self.history,
            stride=self.stride,
            max_future_frames=self.max_future_frames,
        )
        if not self._anchors:
            raise ValueError(
                "No valid windows in {} for history={}, stride={} and max delay={} ms".format(
                    sequence.name, history, stride, max_delay_ms
                )
            )

    def __len__(self) -> int:
        return len(self._anchors)

    def __getitem__(self, index: int) -> Dict[str, Union[torch.Tensor, str]]:
        track_id, anchor_frame = self._anchors[index]
        if self.fixed_delay_ms is None:
            low, high = self.delay_range_ms
            delay_ms = low + (high - low) * torch.rand(()).item()
        else:
            delay_ms = self.fixed_delay_ms

        track = self._tracks[track_id]
        target_box = box_at_time(track, anchor_frame, delay_ms, self.sequence.fps)
        history_frames = list(
            range(anchor_frame - self.history + 1, anchor_frame + 1)
        )
        history_boxes = [track[frame] for frame in history_frames]

        motions = [
            normalized_motion(history_boxes[index], history_boxes[index + 1])
            for index in range(len(history_boxes) - 1)
        ]
        intervals = [
            time_interval(history_frames[index], history_frames[index + 1], self.sequence.fps)
            for index in range(len(history_frames) - 1)
        ]
        speeds = [
            normalized_speed(motion, interval)
            for motion, interval in zip(motions, intervals)
        ]
        target_motion = relative_motion(history_boxes[-1], target_box)
        delay_encoding = fourier_delay_encoding(
            delay_ms, self.max_delay_ms, self.num_frequencies
        )
        target_frame_position = anchor_frame + delay_ms * self.sequence.fps / 1000.0

        return {
            "history_boxes": torch.tensor(history_boxes, dtype=torch.float32),
            "normalized_motion": torch.tensor(motions, dtype=torch.float32).reshape(-1, 2),
            "time_intervals": torch.tensor(intervals, dtype=torch.float32),
            "relative_motion_factor": torch.tensor(speeds, dtype=torch.float32),
            "delay_encoding": torch.tensor(delay_encoding, dtype=torch.float32),
            "target_box": torch.tensor(target_box, dtype=torch.float32),
            "target_motion": torch.tensor(target_motion, dtype=torch.float32),
            "frame_ids": torch.tensor(history_frames, dtype=torch.int64),
            "anchor_frame": torch.tensor(anchor_frame, dtype=torch.int64),
            "target_frame": torch.tensor(
                int(round(target_frame_position)), dtype=torch.int64
            ),
            "target_frame_position": torch.tensor(
                target_frame_position, dtype=torch.float32
            ),
            "target_time_ms": torch.tensor(
                (anchor_frame - 1) * 1000.0 / self.sequence.fps + delay_ms,
                dtype=torch.float32,
            ),
            "delay_ms": torch.tensor(delay_ms, dtype=torch.float32),
            "fps": torch.tensor(self.sequence.fps, dtype=torch.float32),
            "sequence": self.sequence.name,
            "track_id": torch.tensor(track_id, dtype=torch.int64),
        }



def _join_sequences(
    specs: Sequence[SequenceSpec],
    history: int,
    stride: int,
    delay_range_ms: Tuple[float, float],
    fixed_delay_ms: Optional[float],
    max_delay_ms: float,
    num_frequencies: int,
) -> ConcatDataset:
    if not specs:
        raise ValueError("Cannot build a dataset from an empty sequence split")
    datasets = []
    for spec in specs:
        try:
            datasets.append(
                SequenceWindowDataset(
                    load_sequence(spec),
                    history=history,
                    stride=stride,
                    delay_range_ms=delay_range_ms,
                    delay_ms=fixed_delay_ms,
                    max_delay_ms=max_delay_ms,
                    num_frequencies=num_frequencies,
                )
            )
        except ValueError as error:
            if "No valid windows" not in str(error):
                raise
    if not datasets:
        raise ValueError("No sequence produced a valid delayed-motion window")
    return ConcatDataset(datasets)


def build_data_loaders(
    root: str,
    batch_size: int = 128,
    num_workers: int = 4,
    history: int = 8,
    stride: int = 8,
    delay_range_ms: Tuple[float, float] = (33.0, 200.0),
    eval_delay_ms: float = 100.0,
    num_frequencies: int = 8,
    seed: int = DEFAULT_SEED,
    strict: bool = True,
) -> Dict[str, DataLoader]:
    """Build train/validation/test DataLoaders across all available datasets.

    Training samples a delay uniformly from delay_range_ms. Validation and test
    use one fixed delay, and all splits share the maximum-delay anchor set so
    model comparisons see the same windows.
    """
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    if eval_delay_ms > delay_range_ms[1]:
        raise ValueError("eval_delay_ms cannot exceed the maximum training delay")
    split_specs = discover_splits(root, seed=seed, strict=strict)
    max_delay_ms = delay_range_ms[1]
    generator = torch.Generator().manual_seed(seed)
    loaders: Dict[str, DataLoader] = {}
    for split in ("train", "val", "test"):
        fixed_delay = None if split == "train" else eval_delay_ms
        datasets = [
            _join_sequences(
                list(dataset_splits[split].values()),
                history,
                stride,
                delay_range_ms,
                fixed_delay,
                max_delay_ms,
                num_frequencies,
            )
            for dataset_splits in split_specs.values()
            if dataset_splits[split]
        ]
        if not datasets:
            raise ValueError("No sequences available for the {} split".format(split))
        loaders[split] = DataLoader(
            ConcatDataset(datasets),
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(num_workers > 0),
            generator=generator if split == "train" else None,
        )
    return loaders


def build_eval_loaders(
    root: str,
    delays_ms: Sequence[float] = (33.0, 66.0, 100.0, 133.0, 166.0, 200.0),
    split: str = "test",
    batch_size: int = 128,
    num_workers: int = 4,
    history: int = 8,
    stride: int = 8,
    delay_range_ms: Tuple[float, float] = (33.0, 200.0),
    num_frequencies: int = 8,
    seed: int = DEFAULT_SEED,
    strict: bool = True,
) -> Dict[float, DataLoader]:
    """Build one evaluation loader per delay using an identical anchor set."""
    if split not in ("val", "test"):
        raise ValueError("split must be 'val' or 'test'")
    if not delays_ms or any(delay <= 0 for delay in delays_ms):
        raise ValueError("delays_ms must contain positive delays")
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")

    split_specs = discover_splits(root, seed=seed, strict=strict)
    max_delay_ms = max(max(delays_ms), delay_range_ms[1])
    loaders: Dict[float, DataLoader] = {}
    for delay_ms in delays_ms:
        datasets = [
            _join_sequences(
                list(dataset_splits[split].values()),
                history,
                stride,
                delay_range_ms,
                float(delay_ms),
                max_delay_ms,
                num_frequencies,
            )
            for dataset_splits in split_specs.values()
            if dataset_splits[split]
        ]
        if not datasets:
            raise ValueError("No sequences available for the {} split".format(split))
        loaders[float(delay_ms)] = DataLoader(
            ConcatDataset(datasets),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(num_workers > 0),
        )
    return loaders

