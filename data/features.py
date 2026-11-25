"""Pure-Python feature extraction for delay-aware motion prediction."""

import math
from typing import Sequence, Tuple

Box = Tuple[float, float, float, float]
Vector2 = Tuple[float, float]
Vector4 = Tuple[float, float, float, float]


def box_center(box: Sequence[float]) -> Vector2:
    """Return the center of an xywh box."""
    x, y, width, height = box
    return (x + width / 2.0, y + height / 2.0)


def normalized_motion(previous_box: Sequence[float], box: Sequence[float]) -> Vector2:
    """Center displacement normalized by the previous box width and height."""
    previous_cx, previous_cy = box_center(previous_box)
    cx, cy = box_center(box)
    previous_width, previous_height = previous_box[2], previous_box[3]
    if previous_width <= 0 or previous_height <= 0:
        raise ValueError("Cannot normalize motion by a non-positive box size")
    return (
        (cx - previous_cx) / previous_width,
        (cy - previous_cy) / previous_height,
    )


def time_interval(previous_frame: int, frame: int, fps: float) -> float:
    """Time between two frames in seconds."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    return (frame - previous_frame) / fps


def normalized_speed(motion: Sequence[float], interval_seconds: float) -> float:
    """Magnitude of normalized motion per second."""
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")
    return math.hypot(motion[0], motion[1]) / interval_seconds


def relative_motion(last_box: Sequence[float], target_box: Sequence[float]) -> Vector4:
    """Center offset plus log-size ratio, normalized by the last observed box."""
    last_cx, last_cy = box_center(last_box)
    target_cx, target_cy = box_center(target_box)
    last_width, last_height = last_box[2], last_box[3]
    target_width, target_height = target_box[2], target_box[3]
    if last_width <= 0 or last_height <= 0:
        raise ValueError("Cannot normalize motion by a non-positive box size")
    if target_width <= 0 or target_height <= 0:
        raise ValueError("Target box size must be positive")
    return (
        (target_cx - last_cx) / last_width,
        (target_cy - last_cy) / last_height,
        math.log(target_width / last_width),
        math.log(target_height / last_height),
    )


def fourier_delay_encoding(
    delay_ms: float,
    max_delay_ms: float,
    num_frequencies: int = 8,
) -> Tuple[float, ...]:
    """Encode a delay with sin/cos features at logarithmically spaced frequencies.

    The delay is normalized to [0, 1] by the maximum delay so the encoding keeps
    the same frequency range across datasets with different frame rates.
    """
    if max_delay_ms <= 0:
        raise ValueError("max_delay_ms must be positive")
    if num_frequencies < 1:
        raise ValueError("num_frequencies must be at least 1")
    phase = delay_ms / max_delay_ms
    values = []
    for exponent in range(num_frequencies):
        angle = 2.0 * math.pi * (2 ** exponent) * phase
        values.extend((math.sin(angle), math.cos(angle)))
    return tuple(values)
