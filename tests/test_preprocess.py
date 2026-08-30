import unittest

import numpy as np
from PIL import Image

from src.utils.preprocess import clear_transparent_rgb


class PreprocessTests(unittest.TestCase):
    def test_clear_transparent_rgb_preserves_alpha_and_visible_pixels(self):
        source_pixels = np.array(
            [[[10, 20, 30, 0], [40, 50, 60, 1]]],
            dtype=np.uint8,
        )
        source = Image.fromarray(source_pixels)

        output = np.asarray(clear_transparent_rgb(source))

        np.testing.assert_array_equal(
            output,
            np.array([[[0, 0, 0, 0], [40, 50, 60, 1]]], dtype=np.uint8),
        )
        np.testing.assert_array_equal(np.asarray(source), source_pixels)


if __name__ == '__main__':
    unittest.main()
