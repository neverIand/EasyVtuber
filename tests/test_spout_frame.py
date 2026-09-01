import unittest

import numpy as np

from src.utils.spout_frame import (
    create_spout_staging_buffer,
    stage_spout_frame,
)


class SpoutFrameTests(unittest.TestCase):
    def test_staging_preserves_every_channel_and_reuses_destination(self):
        source = np.array(
            [
                [[0, 1, 254, 255], [17, 33, 65, 129]],
                [[255, 128, 64, 32], [9, 8, 7, 6]],
            ],
            dtype=np.uint8,
        )
        destination = create_spout_staging_buffer(source.shape)

        result = stage_spout_frame(source, destination)

        self.assertIs(result, destination)
        self.assertEqual(result.dtype, np.int32)
        self.assertTrue(result.flags.c_contiguous)
        np.testing.assert_array_equal(result, source)

    def test_subsequent_frame_overwrites_the_same_buffer(self):
        destination = create_spout_staging_buffer((2, 3, 4))
        first_pointer = destination.ctypes.data

        stage_spout_frame(
            np.full((2, 3, 4), 11, dtype=np.uint8),
            destination,
        )
        stage_spout_frame(
            np.full((2, 3, 4), 222, dtype=np.uint8),
            destination,
        )

        self.assertEqual(destination.ctypes.data, first_pointer)
        np.testing.assert_array_equal(destination, 222)

    def test_invalid_shapes_and_dtypes_are_rejected(self):
        with self.assertRaises(ValueError):
            create_spout_staging_buffer((512, 512, 3))

        destination = create_spout_staging_buffer((2, 2, 4))
        with self.assertRaises(TypeError):
            stage_spout_frame(
                np.zeros((2, 2, 4), dtype=np.float32),
                destination,
            )
        with self.assertRaises(ValueError):
            stage_spout_frame(
                np.zeros((1, 2, 4), dtype=np.uint8),
                destination,
            )


if __name__ == '__main__':
    unittest.main()
