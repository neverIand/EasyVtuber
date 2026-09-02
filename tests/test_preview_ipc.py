import subprocess
import sys
import unittest

import numpy as np

from src.utils.preview_ipc import PreviewFrameFormatter, PreviewSharedBuffer


class PreviewSharedBufferTests(unittest.TestCase):
    def test_attached_reader_receives_only_new_complete_frames(self):
        owner = PreviewSharedBuffer.create(width=4, height=3)
        reader = PreviewSharedBuffer.attach(owner.name)
        try:
            self.assertIsNone(reader.read_latest())

            first = np.full((3, 4, 4), 17, dtype=np.uint8)
            owner.publish_rgba(first)
            np.testing.assert_array_equal(reader.read_latest(), first)
            self.assertIsNone(reader.read_latest())

            second = np.full((3, 4, 4), 231, dtype=np.uint8)
            owner.publish_rgba(second)
            np.testing.assert_array_equal(reader.read_latest(), second)
        finally:
            reader.close()
            owner.close()

    def test_invalid_frame_is_rejected_without_advancing_reader(self):
        owner = PreviewSharedBuffer.create(width=2, height=2)
        reader = PreviewSharedBuffer.attach(owner.name)
        try:
            with self.assertRaises(ValueError):
                owner.publish_rgba(np.zeros((1, 1, 4), dtype=np.uint8))
            self.assertIsNone(reader.read_latest())
        finally:
            reader.close()
            owner.close()

    def test_separate_process_can_publish_to_the_launcher_owner(self):
        owner = PreviewSharedBuffer.create(width=2, height=2)
        child = r'''
import sys
import numpy as np
from src.utils.preview_ipc import PreviewSharedBuffer

buffer = PreviewSharedBuffer.attach(sys.argv[1])
try:
    buffer.publish_rgba(np.full((2, 2, 4), 73, dtype=np.uint8))
finally:
    buffer.close()
'''
        try:
            result = subprocess.run(
                [sys.executable, '-c', child, owner.name],
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            np.testing.assert_array_equal(
                owner.read_latest(),
                np.full((2, 2, 4), 73, dtype=np.uint8),
            )
        finally:
            owner.close()


class PreviewFrameFormatterTests(unittest.TestCase):
    def test_bgr_is_converted_to_rgba(self):
        source = np.zeros((2, 2, 3), dtype=np.uint8)
        source[:, :] = (7, 11, 19)

        actual = PreviewFrameFormatter(2, 2).format(source, 'BGR')

        np.testing.assert_array_equal(actual[:, :, 0], 19)
        np.testing.assert_array_equal(actual[:, :, 1], 11)
        np.testing.assert_array_equal(actual[:, :, 2], 7)
        np.testing.assert_array_equal(actual[:, :, 3], 255)

    def test_wide_frame_is_letterboxed_without_losing_alpha(self):
        source = np.full((2, 4, 4), 255, dtype=np.uint8)
        source[:, :, 3] = 91

        actual = PreviewFrameFormatter(4, 4).format(source, 'RGBA')

        np.testing.assert_array_equal(actual[0], 0)
        np.testing.assert_array_equal(actual[3], 0)
        np.testing.assert_array_equal(actual[1:3, :, :3], 255)
        np.testing.assert_array_equal(actual[1:3, :, 3], 91)


if __name__ == '__main__':
    unittest.main()
