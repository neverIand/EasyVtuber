import unittest
import sys

import numpy as np

original_argv = sys.argv
try:
    sys.argv = [sys.argv[0]]
    from src.args import args
    from src.utils.pose_simplify import _get_ifm_converter, _get_simplify_config, pose_simplify
finally:
    sys.argv = original_argv


class PoseSimplifyTests(unittest.TestCase):
    def setUp(self):
        self.original_simplify = args.simplify
        self.original_eyebrow = args.eyebrow

    def tearDown(self):
        args.simplify = self.original_simplify
        args.eyebrow = self.original_eyebrow

    def test_quantization_config_is_cached_and_immutable(self):
        first = _get_simplify_config(1)
        second = _get_simplify_config(1)
        converter = _get_ifm_converter()

        self.assertIs(first, second)
        self.assertFalse(first[0].flags.writeable)
        self.assertFalse(first[1].flags.writeable)
        self.assertEqual(first[0][converter.iris_rotation_x_index], 25)

    def test_level_one_matches_scalar_quantization(self):
        args.simplify = 1
        args.eyebrow = True
        model_input = np.linspace(-1.25, 1.25, 45, dtype=np.float32)
        expected_input = model_input.copy()
        scales, quantized = _get_simplify_config(1)
        for index in range(45):
            if quantized[index]:
                expected_input[index] = round(
                    expected_input[index] * scales[index]
                ) / scales[index]

        output = pose_simplify(model_input)

        np.testing.assert_array_equal(model_input, expected_input)
        np.testing.assert_array_equal(output[0], expected_input)


if __name__ == '__main__':
    unittest.main()
