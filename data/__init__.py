"""Dataset discovery, annotation parsing, and PyTorch window datasets."""

from data.annotations import SequenceAnnotations, SequenceSpec, load_sequence
from data.features import (
    fourier_delay_encoding,
    normalized_motion,
    normalized_speed,
    relative_motion,
    time_interval,
)
from data.splits import discover_splits, split_names

__all__ = [
    "SequenceAnnotations",
    "SequenceSpec",
    "SequenceWindowDataset",
    "build_data_loaders",
    "build_eval_loaders",
    "discover_splits",
    "fourier_delay_encoding",
    "load_sequence",
    "normalized_motion",
    "normalized_speed",
    "relative_motion",
    "split_names",
    "time_interval",
]


def __getattr__(name):
    if name in (
        "SequenceWindowDataset",
        "build_data_loaders",
        "build_eval_loaders",
    ):
        from data.datasets import (
            SequenceWindowDataset,
            build_data_loaders,
            build_eval_loaders,
        )

        return {
            "SequenceWindowDataset": SequenceWindowDataset,
            "build_data_loaders": build_data_loaders,
            "build_eval_loaders": build_eval_loaders,
        }[name]
    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))
