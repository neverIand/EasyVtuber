import numpy as np

from tha2.mocap.ifacialmocap_constants import (
    HEAD_BONE_QUAT,
    HEAD_BONE_X,
    HEAD_BONE_Y,
    HEAD_BONE_Z,
    LEFT_EYE_BONE_X,
    LEFT_EYE_BONE_Y,
    LEFT_EYE_BONE_Z,
    RIGHT_EYE_BONE_X,
    RIGHT_EYE_BONE_Y,
    RIGHT_EYE_BONE_Z,
)


DEGREES_PER_RADIAN = 57.3


def parse_ifacialmocap_v1_pose(payload: str):
    """Parse the legacy iFacialMocap UDP payload used by EasyVtuber."""
    data = {}

    for raw_item in payload.split('|'):
        item = raw_item.strip()
        if not item:
            continue

        key, separator, raw_value = item.partition('#')
        if separator:
            normalized_key = key.replace('_L', 'Left').replace('_R', 'Right')
            data[normalized_key] = [float(value) for value in raw_value.split(',')]
            continue

        key, separator, raw_value = item.partition('-')
        if separator:
            normalized_key = key.replace('_L', 'Left').replace('_R', 'Right')
            data[normalized_key] = float(raw_value) / 100

    head = data['=head']
    right_eye = data['rightEye']
    left_eye = data['leftEye']

    data[HEAD_BONE_X] = head[0] / DEGREES_PER_RADIAN
    data[HEAD_BONE_Y] = head[1] / DEGREES_PER_RADIAN
    data[HEAD_BONE_Z] = head[2] / DEGREES_PER_RADIAN
    data[HEAD_BONE_QUAT] = [head[3], head[4], head[5], 1]
    data[RIGHT_EYE_BONE_X] = right_eye[0] / DEGREES_PER_RADIAN
    data[RIGHT_EYE_BONE_Y] = right_eye[1] / DEGREES_PER_RADIAN
    data[RIGHT_EYE_BONE_Z] = right_eye[2] / DEGREES_PER_RADIAN
    data[LEFT_EYE_BONE_X] = left_eye[0] / DEGREES_PER_RADIAN
    data[LEFT_EYE_BONE_Y] = left_eye[1] / DEGREES_PER_RADIAN
    data[LEFT_EYE_BONE_Z] = left_eye[2] / DEGREES_PER_RADIAN
    return data


def fill_ifacialmocap_model_input(
    converted_pose,
    breath_value: float,
    output: np.ndarray,
) -> np.ndarray:
    """Fill the 45-value model pose without per-frame intermediate lists."""
    output[:42] = converted_pose[:42]
    output[26] *= 1.5
    output[42] = converted_pose[40]
    output[43] = converted_pose[41]
    output[44] = breath_value
    return output
