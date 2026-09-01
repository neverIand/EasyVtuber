import unittest

import cv2
import numpy as np

from src.utils.frame_transform import apply_output_transform, build_output_transform


class FrameTransformTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.arange(8 * 10 * 4, dtype=np.uint8).reshape(8, 10, 4)

    def test_default_transform_uses_zero_copy_identity_path(self):
        transform = build_output_transform(
            np.zeros(4, dtype=np.float32),
            self.frame.shape,
            extend_movement=False,
            bongo=False,
        )

        self.assertIsNone(transform)
        self.assertIs(apply_output_transform(self.frame, transform), self.frame)

    def test_identity_path_can_copy_for_debug_overlay(self):
        output = apply_output_transform(self.frame, None, copy_identity=True)

        self.assertIsNot(output, self.frame)
        np.testing.assert_array_equal(output, self.frame)

    def test_non_identity_transform_matches_legacy_calculation(self):
        position = np.array([0.08, -0.03, 0.04, 1.0], dtype=np.float32)
        transform = build_output_transform(
            position,
            self.frame.shape,
            extend_movement=True,
            bongo=True,
        )

        scale = position[2] + 1
        angle = -position[0] * 10 - 5
        expected_matrix = cv2.getRotationMatrix2D((5, 4), angle, scale)
        expected_matrix[0, 2] += position[0] * 400 * scale
        expected_matrix[1, 2] += -position[1] * 600 * scale
        expected = cv2.warpAffine(self.frame, expected_matrix, (10, 8))

        np.testing.assert_allclose(transform, expected_matrix)
        np.testing.assert_array_equal(
            apply_output_transform(self.frame, transform),
            expected,
        )

    def test_non_identity_transform_can_reuse_destination(self):
        transform = build_output_transform(
            np.array([0.08, -0.03, 0.04, 1.0], dtype=np.float32),
            self.frame.shape,
            extend_movement=True,
            bongo=True,
        )
        destination = np.empty_like(self.frame)
        expected = apply_output_transform(self.frame, transform)

        actual = apply_output_transform(self.frame, transform, dst=destination)

        self.assertIs(actual, destination)
        np.testing.assert_array_equal(actual, expected)

    def test_identity_transform_can_copy_into_destination(self):
        destination = np.empty_like(self.frame)

        actual = apply_output_transform(self.frame, None, dst=destination)

        self.assertIs(actual, destination)
        np.testing.assert_array_equal(actual, self.frame)


if __name__ == '__main__':
    unittest.main()
