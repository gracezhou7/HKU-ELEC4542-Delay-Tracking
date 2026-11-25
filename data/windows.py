"""Window indexing and time-based box interpolation."""

import math
from typing import List, Mapping, Tuple

Box = Tuple[float, float, float, float]


def delay_to_future_frames(delay_ms: float, fps: float) -> int:
    """Number of future frames needed to interpolate a box after delay_ms."""
    if delay_ms <= 0 or fps <= 0:
        raise ValueError("delay_ms and fps must be positive")
    return max(1, int(math.ceil(delay_ms * fps / 1000.0)))


def build_anchors(
    tracks: Mapping[int, Mapping[int, Box]],
    history: int,
    stride: int,
    max_future_frames: int,
) -> List[Tuple[int, int]]:
    """Return (track_id, anchor_frame) pairs with contiguous history and future.

    The anchor is the last observed frame. Windows are only emitted when all
    history frames and all future frames up to max_future_frames are present.
    """
    if history < 1:
        raise ValueError("history must be at least 1")
    if stride < 1:
        raise ValueError("stride must be at least 1")
    if max_future_frames < 1:
        raise ValueError("max_future_frames must be at least 1")

    anchors: List[Tuple[int, int]] = []
    for track_id, track in sorted(tracks.items()):
        frame_ids = sorted(track)
        last_start = len(frame_ids) - max_future_frames
        for anchor_index in range(history - 1, last_start, stride):
            anchor_frame = frame_ids[anchor_index]
            history_ids = frame_ids[anchor_index - history + 1 : anchor_index + 1]
            future_ids = frame_ids[
                anchor_index + 1 : anchor_index + max_future_frames + 1
            ]
            expected_history = list(range(anchor_frame - history + 1, anchor_frame + 1))
            expected_future = list(
                range(anchor_frame + 1, anchor_frame + max_future_frames + 1)
            )
            if history_ids == expected_history and future_ids == expected_future:
                anchors.append((track_id, anchor_frame))
    return anchors


def box_at_time(
    track: Mapping[int, Box],
    anchor_frame: int,
    delay_ms: float,
    fps: float,
) -> Box:
    """Linearly interpolate the target box at anchor time plus the requested delay."""
    if fps <= 0 or delay_ms < 0:
        raise ValueError("fps must be positive and delay_ms non-negative")
    frame_position = anchor_frame + delay_ms * fps / 1000.0
    lower = int(math.floor(frame_position))
    upper = int(math.ceil(frame_position))
    if lower == upper:
        return track[lower]
    if lower not in track or upper not in track:
        raise KeyError(
            "Missing frame {} or {} for interpolation".format(lower, upper)
        )
    alpha = frame_position - lower
    lower_box = track[lower]
    upper_box = track[upper]
    return tuple(
        (1.0 - alpha) * start + alpha * end
        for start, end in zip(lower_box, upper_box)
    )
