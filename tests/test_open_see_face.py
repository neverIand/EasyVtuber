import struct
import unittest

from src.utils.open_see_face import parse_open_see_face_packet


PACKET = struct.Struct('=di2f2fB1f4f3f3f68f136f210f14f')


class OpenSeeFaceTests(unittest.TestCase):
    def test_parser_extracts_consumed_values(self):
        values = list(PACKET.unpack(bytes(PACKET.size)))
        values[4] = 0.75
        values[5] = 0.65
        values[12:18] = [20.0, 15.0, 100.0, 2.0, 3.0, 4.0]
        values[420:432] = [
            1.0, 2.0, 3.0, 4.0, 6.0, 8.0,
            0.5, 1.0, 1.5, 1.5, 2.0, 2.5,
        ]
        values[435] = 0.25
        values[438] = 0.35
        values[444] = 0.45

        frame = parse_open_see_face_packet(PACKET.pack(*values))

        self.assertAlmostEqual(frame.left_eye_open, 0.65)
        self.assertAlmostEqual(frame.right_eye_open, 0.75)
        self.assertAlmostEqual(frame.rotation_x, 160.0)
        self.assertAlmostEqual(frame.rotation_y, 5.0)
        self.assertAlmostEqual(frame.rotation_z, 10.0)
        self.assertAlmostEqual(frame.translation_x, 3.0)
        self.assertAlmostEqual(frame.translation_y, -2.0)
        self.assertAlmostEqual(frame.translation_z, -4.0)
        self.assertAlmostEqual(frame.eye_rotation_x, 3.0 / (83.0 ** 0.5))
        self.assertAlmostEqual(frame.eye_rotation_y, 5.0 / (83.0 ** 0.5))
        self.assertAlmostEqual(frame.eyebrow_left, 0.25)
        self.assertAlmostEqual(frame.eyebrow_right, 0.35)
        self.assertAlmostEqual(frame.mouth_open, 0.45)


if __name__ == '__main__':
    unittest.main()
