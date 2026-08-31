import unittest

import cv2
import numpy as np

from src.utils.camera_frame import prepare_mediapipe_rgb


class CameraFrameTests(unittest.TestCase):
    def test_conversion_matches_opencv_and_reuses_buffer(self):
        frame = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
        expected = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        first = prepare_mediapipe_rgb(frame)
        second = prepare_mediapipe_rgb(frame, first)

        self.assertIs(first, second)
        self.assertFalse(second.flags.writeable)
        np.testing.assert_array_equal(second, expected)

    def test_shape_change_replaces_buffer(self):
        first = prepare_mediapipe_rgb(np.zeros((2, 3, 3), dtype=np.uint8))
        second = prepare_mediapipe_rgb(
            np.zeros((4, 5, 3), dtype=np.uint8),
            first,
        )

        self.assertIsNot(first, second)
        self.assertEqual(second.shape, (4, 5, 3))
        self.assertFalse(second.flags.writeable)


if __name__ == '__main__':
    unittest.main()
