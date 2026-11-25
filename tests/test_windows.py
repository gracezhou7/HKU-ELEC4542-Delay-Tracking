import unittest

from data.windows import box_at_time, build_anchors, delay_to_future_frames


class WindowTests(unittest.TestCase):
    def test_delay_to_future_frames_rounds_up(self):
        self.assertEqual(delay_to_future_frames(200.0, 30.0), 6)
        self.assertEqual(delay_to_future_frames(33.0, 30.0), 1)
        self.assertEqual(delay_to_future_frames(33.0, 14.0), 1)

    def test_anchors_require_contiguous_history_and_future(self):
        track = {frame: (frame, 0, 10, 10) for frame in range(1, 31)}
        anchors = build_anchors({1: track}, history=8, stride=8, max_future_frames=6)
        self.assertEqual(anchors, [(1, 8), (1, 16), (1, 24)])

    def test_gap_removes_invalid_anchors(self):
        track = {frame: (frame, 0, 10, 10) for frame in range(1, 31) if frame != 10}
        anchors = build_anchors({1: track}, history=8, stride=1, max_future_frames=6)
        self.assertNotIn((1, 8), anchors)
        self.assertIn((1, 18), anchors)

    def test_box_interpolation_handles_sub_frame_delays(self):
        track = {1: (0, 0, 10, 10), 2: (2, 0, 10, 10)}
        self.assertEqual(box_at_time(track, 1, 50.0, 20.0), (2.0, 0.0, 10.0, 10.0))
        self.assertEqual(box_at_time(track, 1, 50.0, 10.0), (1.0, 0.0, 10.0, 10.0))


if __name__ == "__main__":
    unittest.main()
