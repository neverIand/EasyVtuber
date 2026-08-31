import struct
from typing import NamedTuple

import numpy as np


_OPEN_SEE_FACE_PACKET = struct.Struct('=di2f2fB1f4f3f3f68f136f210f14f')


class OpenSeeFaceFrame(NamedTuple):
    left_eye_open: float
    right_eye_open: float
    rotation_x: float
    rotation_y: float
    rotation_z: float
    translation_x: float
    translation_y: float
    translation_z: float
    eye_rotation_x: float
    eye_rotation_y: float
    eyebrow_left: float
    eyebrow_right: float
    mouth_open: float


def parse_open_see_face_packet(packet: bytes) -> OpenSeeFaceFrame:
    """Extract only the OpenSeeFace values consumed by EasyVtuber."""
    raw = _OPEN_SEE_FACE_PACKET.unpack(packet)
    eye = np.array(
        [
            raw[420] - raw[426] + raw[423] - raw[429],
            raw[421] - raw[427] + raw[424] - raw[430],
            raw[422] - raw[428] + raw[425] - raw[431],
        ]
    )
    eye /= np.linalg.norm(eye)

    return OpenSeeFaceFrame(
        left_eye_open=raw[5],
        right_eye_open=raw[4],
        rotation_x=(-raw[12] + 360) % 360 - 180,
        rotation_y=raw[13] - 10,
        rotation_z=raw[14] - 90,
        translation_x=raw[16],
        translation_y=-raw[15],
        translation_z=-raw[17],
        eye_rotation_x=eye[0],
        eye_rotation_y=eye[1],
        eyebrow_left=raw[435],
        eyebrow_right=raw[438],
        mouth_open=raw[444],
    )
