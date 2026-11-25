import tempfile
import unittest
from pathlib import Path

from data.annotations import SequenceSpec, load_sequence


class AnnotationLoadingTests(unittest.TestCase):
    def test_filters_invalid_invisible_non_pedestrian_and_outside_boxes(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            sequence_dir = Path(temporary_directory) / "seq"
            ground_truth_dir = sequence_dir / "gt"
            ground_truth_dir.mkdir(parents=True)
            (sequence_dir / "seqinfo.ini").write_text(
                "[Sequence]\nframeRate=30\nimWidth=100\nimHeight=80\n",
                encoding="utf-8",
            )
            annotation_path = ground_truth_dir / "gt.txt"
            annotation_path.write_text(
                "1,1,10,10,20,30,1,1,1\n"
                "1,2,10,10,20,30,0,1,1\n"
                "1,3,10,10,20,30,1,2,1\n"
                "1,4,10,10,20,30,1,1,0\n"
                "1,5,120,10,20,30,1,1,1\n"
                "1,6,-30,10,20,30,1,1,1\n",
                encoding="utf-8",
            )

            sequence = load_sequence(SequenceSpec("seq", annotation_path, 30.0))

            self.assertEqual(sequence.frames, {1: {1: (10.0, 10.0, 20.0, 30.0)}})
            self.assertEqual(sequence.image_width, 100)
            self.assertEqual(sequence.image_height, 80)

    def test_loads_otb_ground_truth_with_invalid_rows(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            sequence_dir = Path(temporary_directory) / "DTB70" / "Animal1"
            sequence_dir.mkdir(parents=True)
            annotation_path = sequence_dir / "groundtruth_rect.txt"
            annotation_path.write_text(
                "0,0,10,20\n"
                "\n"
                "2 3 10 20\n"
                "0,0,0,0\n",
                encoding="utf-8",
            )

            sequence = load_sequence(
                SequenceSpec("Animal1", annotation_path, 30.0, format="otb")
            )

            self.assertEqual(
                sequence.frames,
                {1: {1: (0.0, 0.0, 10.0, 20.0)}, 3: {1: (2.0, 3.0, 10.0, 20.0)}},
            )


if __name__ == "__main__":
    unittest.main()
