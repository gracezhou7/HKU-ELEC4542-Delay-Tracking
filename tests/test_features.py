import math
import unittest

from data.features import (
    box_center,
    fourier_delay_encoding,
    normalized_motion,
    normalized_speed,
    relative_motion,
    time_interval,
)


class FeatureTests(unittest.TestCase):
    def test_normalized_motion_uses_previous_box_size(self):
        self.assertEqual(box_center((0, 0, 10, 20)), (5, 10))
        self.assertEqual(
            normalized_motion((0, 0, 10, 20), (2, 4, 10, 20)),
            (0.2, 0.2),
        )

    def test_time_interval_is_in_seconds(self):
        self.assertAlmostEqual(time_interval(1, 2, 30.0), 1.0 / 30.0)
        self.assertAlmostEqual(time_interval(1, 4, 25.0), 0.12)

    def test_relative_motion_uses_center_and_log_size(self):
        motion = relative_motion((0, 0, 10, 20), (2, 4, 20, 40))
        self.assertAlmostEqual(motion[0], 0.7)
        self.assertAlmostEqual(motion[1], 0.7)
        self.assertAlmostEqual(motion[2], math.log(2))
        self.assertAlmostEqual(motion[3], math.log(2))

    def test_normalized_speed(self):
        self.assertAlmostEqual(normalized_speed((0.3, 0.4), 0.5), 1.0)

    def test_fourier_encoding_is_bounded_and_has_expected_length(self):
        encoding = fourier_delay_encoding(66.0, 200.0, num_frequencies=4)
        self.assertEqual(len(encoding), 8)
        self.assertTrue(all(-1.0 <= value <= 1.0 for value in encoding))
        self.assertEqual(
            fourier_delay_encoding(66.0, 200.0, 4),
            fourier_delay_encoding(66.0, 200.0, 4),
        )


if __name__ == "__main__":
    unittest.main()
