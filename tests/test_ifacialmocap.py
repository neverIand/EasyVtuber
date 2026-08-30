import unittest

import numpy as np
import scipy.optimize

from src.utils.ifacialmocap import (
    DEGREES_PER_RADIAN,
    fill_ifacialmocap_model_input,
    parse_ifacialmocap_v1_pose,
)
from tha2.mocap.ifacialmocap_constants import (
    BLENDSHAPE_NAMES,
    EYE_BLINK_LEFT,
    HEAD_BONE_QUAT,
    HEAD_BONE_X,
    LEFT_EYE_BONE_X,
    RIGHT_EYE_BONE_X,
    RIGHT_EYE_BONE_Y,
    RIGHT_EYE_BONE_Z,
)
from tha2.poser.modes.mode_20_wx import IFacialMocapPoseConverter20, _decompose_mouth


def _protocol_name(name):
    if name.endswith('Left'):
        return name[:-4] + '_L'
    if name.endswith('Right'):
        return name[:-5] + '_R'
    return name


class IFacialMocapTests(unittest.TestCase):
    def test_complete_packet_converts_to_model_input(self):
        fields = [
            '{}-{}'.format(_protocol_name(name), (index * 17 + 11) % 75)
            for index, name in enumerate(BLENDSHAPE_NAMES)
        ]
        fields.extend(
            [
                '=head#1.5,-2.0,3.25,0.10,-0.20,0.30',
                'rightEye#1.0,2.0,3.0',
                'leftEye#-1.0,-2.0,-3.0',
            ]
        )

        parsed = parse_ifacialmocap_v1_pose('|'.join(fields))
        converted = IFacialMocapPoseConverter20().convert(parsed)
        output = fill_ifacialmocap_model_input(
            converted,
            breath_value=0.0,
            output=np.empty(45, dtype=np.float32),
        )

        self.assertEqual(len(converted), 45)
        self.assertEqual(output.shape, (45,))
        self.assertTrue(np.isfinite(output).all())

    def test_parser_handles_bones_blendshapes_and_negative_values(self):
        payload = (
            'eyeBlink_L-25|jawOpen--5|=head#1.5,-2,3.25,0.1,-0.2,0.3|'
            'rightEye#1,2,3|leftEye#-1,-2,-3'
        )

        parsed = parse_ifacialmocap_v1_pose(payload)

        self.assertEqual(parsed[EYE_BLINK_LEFT], 0.25)
        self.assertEqual(parsed['jawOpen'], -0.05)
        self.assertAlmostEqual(parsed[HEAD_BONE_X], 1.5 / DEGREES_PER_RADIAN)
        self.assertEqual(parsed[HEAD_BONE_QUAT], [0.1, -0.2, 0.3, 1])
        self.assertAlmostEqual(parsed[RIGHT_EYE_BONE_X], 1 / DEGREES_PER_RADIAN)
        self.assertAlmostEqual(parsed[RIGHT_EYE_BONE_Y], 2 / DEGREES_PER_RADIAN)
        self.assertAlmostEqual(parsed[RIGHT_EYE_BONE_Z], 3 / DEGREES_PER_RADIAN)
        self.assertAlmostEqual(parsed[LEFT_EYE_BONE_X], -1 / DEGREES_PER_RADIAN)

    def test_model_input_fill_matches_legacy_layout(self):
        converted = np.arange(45, dtype=np.float32) / 10
        output = np.empty(45, dtype=np.float32)

        result = fill_ifacialmocap_model_input(converted, 0.25, output)

        expected = converted.copy()
        expected[26] *= 1.5
        expected[42] = converted[40]
        expected[43] = converted[41]
        expected[44] = 0.25
        self.assertIs(result, output)
        np.testing.assert_array_equal(result, expected)

    def test_mouth_decomposition_reuses_identical_states(self):
        _decompose_mouth.cache_clear()

        first = _decompose_mouth(0.6, 0.4, 0.2, 0.3)
        after_first = _decompose_mouth.cache_info()
        second = _decompose_mouth(0.6, 0.4, 0.2, 0.3)
        after_second = _decompose_mouth.cache_info()

        self.assertEqual(first, second)
        self.assertEqual(after_first.misses, 1)
        self.assertEqual(after_second.hits, 1)

    def test_mouth_decomposition_cache_miss_matches_legacy_solver(self):
        mouth_point = np.array([0.61, 0.43, 0.27, 0.38])
        matrix = np.array(
            [
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.5, 0.3, 0.25, 0.75],
                [1.0, 0.5, 0.5, 0.4],
            ]
        )

        def legacy_loss(decomposition):
            return np.linalg.norm(decomposition @ matrix - mouth_point) + 0.01 * np.linalg.norm(
                decomposition,
                ord=1,
            )

        expected = scipy.optimize.minimize(
            legacy_loss,
            np.array([0, 0, 0, 0]),
            bounds=[(0.0, 1.0)] * 4,
        )["x"]
        _decompose_mouth.cache_clear()
        actual = _decompose_mouth(*mouth_point)

        np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
