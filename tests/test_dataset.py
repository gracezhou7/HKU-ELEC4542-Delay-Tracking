import tempfile
import unittest
from pathlib import Path

import torch

from data.annotations import SequenceAnnotations, SequenceSpec, load_sequence
from data.datasets import SequenceWindowDataset


def synthetic_sequence(frame_count=30):
    frames = {
        frame: {1: (float(frame), 0.0, 10.0, 10.0)}
        for frame in range(1, frame_count + 1)
    }
    return SequenceAnnotations(name="synthetic", fps=30.0, frames=frames)


class DatasetTests(unittest.TestCase):
    def test_fixed_delay_outputs_expected_shapes_and_interpolation(self):
        dataset = SequenceWindowDataset(
            synthetic_sequence(),
            history=8,
            stride=8,
            delay_ms=100.0,
            max_delay_ms=200.0,
        )
        self.assertEqual(len(dataset), 3)

        sample = dataset[0]

        self.assertEqual(sample["history_boxes"].shape, (8, 4))
        self.assertEqual(sample["normalized_motion"].shape, (7, 2))
        self.assertEqual(sample["time_intervals"].shape, (7,))
        self.assertEqual(sample["relative_motion_factor"].shape, (7,))
        self.assertEqual(sample["delay_encoding"].shape, (16,))
        self.assertEqual(sample["target_motion"].shape, (4,))
        self.assertAlmostEqual(sample["delay_ms"].item(), 100.0)
        self.assertAlmostEqual(sample["target_box"][0].item(), 11.0)
        self.assertAlmostEqual(sample["target_motion"][0].item(), 0.3)
        self.assertAlmostEqual(sample["time_intervals"][0].item(), 1.0 / 30.0)
        self.assertAlmostEqual(sample["relative_motion_factor"][0].item(), 3.0)
        self.assertTrue(torch.isfinite(sample["target_motion"]).all())

    def test_training_delay_is_sampled_inside_the_range(self):
        dataset = SequenceWindowDataset(
            synthetic_sequence(),
            history=8,
            stride=8,
            delay_range_ms=(33.0, 200.0),
        )
        for index in range(len(dataset)):
            delay = dataset[index]["delay_ms"].item()
            self.assertGreaterEqual(delay, 33.0)
            self.assertLessEqual(delay, 200.0)

    def test_evaluation_delays_share_the_same_anchor_count(self):
        lengths = []
        for delay in (33.0, 100.0, 200.0):
            dataset = SequenceWindowDataset(
                synthetic_sequence(),
                history=8,
                stride=8,
                delay_ms=delay,
                max_delay_ms=200.0,
            )
            lengths.append(len(dataset))
        self.assertEqual(lengths, [3, 3, 3])

    def test_otb_sequence_can_feed_the_window_dataset(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            sequence_dir = Path(temporary_directory) / "Animal1"
            sequence_dir.mkdir(parents=True)
            annotation_path = sequence_dir / "groundtruth_rect.txt"
            annotation_path.write_text(
                "".join(
                    "{},0,10,10\n".format(frame) for frame in range(1, 31)
                ),
                encoding="utf-8",
            )
            sequence = load_sequence(
                SequenceSpec("Animal1", annotation_path, 30.0, format="otb")
            )

            dataset = SequenceWindowDataset(
                sequence,
                history=8,
                stride=8,
                delay_ms=100.0,
                max_delay_ms=200.0,
            )

            self.assertEqual(len(dataset), 3)
            self.assertEqual(dataset[0]["history_boxes"].shape, (8, 4))


if __name__ == "__main__":
    unittest.main()
