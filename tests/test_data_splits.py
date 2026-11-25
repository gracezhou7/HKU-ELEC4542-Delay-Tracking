import tempfile
import unittest
from pathlib import Path

from data.splits import discover_splits, split_names


class SplitNamesTests(unittest.TestCase):
    def test_splits_are_deterministic_disjoint_and_complete(self):
        names = ["seq-{:02d}".format(index) for index in range(70)]
        counts = {"train": 50, "val": 10, "test": 10}

        first = split_names(names, counts, seed=4542)
        second = split_names(reversed(names), counts, seed=4542)

        self.assertEqual(first, second)
        self.assertEqual({key: len(value) for key, value in first.items()}, counts)
        self.assertEqual(len(set(first["train"] + first["val"] + first["test"])), 70)

    def test_rejects_mismatched_counts(self):
        with self.assertRaises(ValueError):
            split_names(["a", "b"], {"train": 1, "val": 1, "test": 1})

    def test_discovers_project_splits_and_uses_mot17_metadata(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self._make_sequences(root / "DanceTrack" / "train", "dance_train", 40)
            self._make_sequences(root / "DanceTrack" / "val", "dance_val", 25)
            self._make_sequences(root / "DTB70", "dtb", 70)
            for index in range(7):
                name = "MOT17-{:02d}".format(index + 1)
                for detector in ("FRCNN", "DPM", "SDP"):
                    sequence_dir = root / "MOT17" / "train" / "{}-{}".format(name, detector)
                    self._make_sequence(sequence_dir)
                    (sequence_dir / "seqinfo.ini").write_text(
                        "[Sequence]\nframeRate={}\n".format(
                            14 if index == 0 else (25 if index == 1 else 30)
                        ),
                        encoding="utf-8",
                    )

            splits = discover_splits(root)

            self.assertEqual(
                {split: len(items) for split, items in splits["DanceTrack"].items()},
                {"train": 40, "val": 10, "test": 15},
            )
            self.assertEqual(
                {split: len(items) for split, items in splits["MOT17"].items()},
                {"train": 5, "val": 1, "test": 1},
            )
            self.assertEqual(
                {split: len(items) for split, items in splits["DTB70"].items()},
                {"train": 50, "val": 10, "test": 10},
            )
            mot_specs = {
                name: spec
                for items in splits["MOT17"].values()
                for name, spec in items.items()
            }
            self.assertTrue(all(spec.annotation_path.parent.parent.name.endswith("-FRCNN")
                                for spec in mot_specs.values()))
            self.assertEqual(sorted(spec.fps for spec in mot_specs.values())[:2], [14, 25])

    def test_falls_back_to_mot17labels_and_skips_missing_datasets(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for index in range(7):
                name = "MOT17-{:02d}".format(index + 1)
                for detector in ("FRCNN", "DPM", "SDP"):
                    sequence_dir = (
                        root / "MOT17Labels" / "train" / "{}-{}".format(name, detector)
                    )
                    self._make_sequence(sequence_dir)
                    (sequence_dir / "seqinfo.ini").write_text(
                        "[Sequence]\nframeRate=30\n", encoding="utf-8"
                    )

            splits = discover_splits(root, strict=False)

            self.assertEqual(set(splits), {"MOT17"})
            self.assertEqual(
                {split: len(items) for split, items in splits["MOT17"].items()},
                {"train": 5, "val": 1, "test": 1},
            )

    def test_discovers_dtb70_otb_annotations(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for index in range(70):
                sequence_dir = root / "DTB70" / "seq_{:02d}".format(index)
                sequence_dir.mkdir(parents=True)
                (sequence_dir / "groundtruth_rect.txt").write_text(
                    "0,0,10,10\n1,0,10,10\n",
                    encoding="utf-8",
                )

            splits = discover_splits(root, strict=False)

            self.assertEqual(set(splits), {"DTB70"})
            self.assertEqual(
                {split: len(items) for split, items in splits["DTB70"].items()},
                {"train": 50, "val": 10, "test": 10},
            )
            self.assertTrue(
                all(
                    spec.format == "otb"
                    for items in splits["DTB70"].values()
                    for spec in items.values()
                )
            )

    @staticmethod
    def _make_sequences(parent, prefix, count):
        for index in range(count):
            SplitNamesTests._make_sequence(
                parent / "{}_{:02d}".format(prefix, index)
            )

    @staticmethod
    def _make_sequence(sequence_dir):
        ground_truth_dir = sequence_dir / "gt"
        ground_truth_dir.mkdir(parents=True)
        (ground_truth_dir / "gt.txt").write_text("1,1,1,1,10,10,1,1,1\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
