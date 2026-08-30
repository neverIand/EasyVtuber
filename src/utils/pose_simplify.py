from ..args import args
import numpy as np
from functools import lru_cache

# Delay import to avoid loading torch during module initialization
ifm_converter = None

def _get_ifm_converter():
    global ifm_converter
    if ifm_converter is None:
        import tha2.poser.modes.mode_20_wx
        ifm_converter = tha2.poser.modes.mode_20_wx.IFacialMocapPoseConverter20()
    return ifm_converter


@lru_cache(maxsize=None)
def _get_simplify_config(simplify):
    """Build immutable quantization data once for each simplify level."""
    converter = _get_ifm_converter()
    scales = np.full(converter.pose_size, 1000, dtype=np.float32)

    if simplify >= 1:
        scales.fill(200)
        scales[converter.eye_wink_left_index] = 50
        scales[converter.eye_wink_right_index] = 50
        scales[converter.eye_happy_wink_left_index] = 50
        scales[converter.eye_happy_wink_right_index] = 50
        scales[converter.eye_surprised_left_index] = 30
        scales[converter.eye_surprised_right_index] = 30
        scales[converter.iris_rotation_x_index] = 25
        scales[converter.iris_rotation_y_index] = 25
        scales[converter.eye_raised_lower_eyelid_left_index] = 10
        scales[converter.eye_raised_lower_eyelid_right_index] = 10
        scales[converter.mouth_lowered_corner_left_index] = 5
        scales[converter.mouth_lowered_corner_right_index] = 5
        scales[converter.mouth_raised_corner_left_index] = 5
        scales[converter.mouth_raised_corner_right_index] = 5
    if simplify >= 2:
        scales[converter.head_x_index] = 100
        scales[converter.head_y_index] = 100
        scales[converter.eye_surprised_left_index] = 10
        scales[converter.eye_surprised_right_index] = 10
        scales[converter.mouth_lowered_corner_left_index] = 0
        scales[converter.mouth_lowered_corner_right_index] = 0
        scales[converter.mouth_raised_corner_left_index] = 0
        scales[converter.mouth_raised_corner_right_index] = 0
    if simplify >= 3:
        scales[converter.iris_rotation_x_index] = 20
        scales[converter.iris_rotation_y_index] = 20
        scales[converter.eye_wink_left_index] = 32
        scales[converter.eye_wink_right_index] = 32
        scales[converter.eye_happy_wink_left_index] = 32
        scales[converter.eye_happy_wink_right_index] = 32
    if simplify >= 4:
        scales[converter.head_x_index] = 50
        scales[converter.head_y_index] = 50
        scales[converter.neck_z_index] = 100
        scales[converter.iris_rotation_x_index] = 10
        scales[converter.iris_rotation_y_index] = 10
        scales[converter.eye_wink_left_index] = 24
        scales[converter.eye_wink_right_index] = 24
        scales[converter.eye_happy_wink_left_index] = 24
        scales[converter.eye_happy_wink_right_index] = 24
        scales[converter.eye_surprised_left_index] = 8
        scales[converter.eye_surprised_right_index] = 8
    for _ in range(4, simplify):
        scales = np.maximum(np.ceil(scales * 0.8), 5)

    quantized = scales > 0
    scales.flags.writeable = False
    quantized.flags.writeable = False
    return scales, quantized


def pose_simplify(model_input):
    converter = _get_ifm_converter()
    simplify = args.simplify

    if args.simplify >= 2:
        model_input[converter.eye_wink_left_index] += model_input[
            converter.eye_happy_wink_left_index
        ]
        model_input[converter.eye_happy_wink_left_index] = (
            model_input[converter.eye_wink_left_index] / 2
        )
        model_input[converter.eye_wink_left_index] = (
            model_input[converter.eye_wink_left_index] / 2
        )
        model_input[converter.eye_wink_right_index] += model_input[
            converter.eye_happy_wink_right_index
        ]
        model_input[converter.eye_happy_wink_right_index] = (
            model_input[converter.eye_wink_right_index] / 2
        )
        model_input[converter.eye_wink_right_index] = (
            model_input[converter.eye_wink_right_index] / 2
        )

        uosum = (
            model_input[converter.mouth_uuu_index]
            + model_input[converter.mouth_ooo_index]
        )
        model_input[converter.mouth_ooo_index] = uosum
        model_input[converter.mouth_uuu_index] = 0
        is_open = (
            model_input[converter.mouth_aaa_index]
            + model_input[converter.mouth_iii_index]
            + uosum
        ) > 0
        model_input[converter.mouth_lowered_corner_left_index] = 0
        model_input[converter.mouth_lowered_corner_right_index] = 0
        model_input[converter.mouth_raised_corner_left_index] = 0.5 if is_open else 0
        model_input[converter.mouth_raised_corner_right_index] = 0.5 if is_open else 0

    if simplify >= 4:
        model_input[converter.eye_raised_lower_eyelid_left_index] = 0
        model_input[converter.eye_raised_lower_eyelid_right_index] = 0
        model_input[converter.eye_wink_left_index] += model_input[
            converter.eye_wink_right_index
        ]
        model_input[converter.eye_wink_right_index] = (
            model_input[converter.eye_wink_left_index] / 2
        )
        model_input[converter.eye_wink_left_index] = (
            model_input[converter.eye_wink_left_index] / 2
        )

        model_input[converter.eye_surprised_left_index] += model_input[
            converter.eye_surprised_right_index
        ]
        model_input[converter.eye_surprised_right_index] = (
            model_input[converter.eye_surprised_left_index] / 2
        )
        model_input[converter.eye_surprised_left_index] = (
            model_input[converter.eye_surprised_left_index] / 2
        )

        model_input[converter.eye_happy_wink_left_index] += model_input[
            converter.eye_happy_wink_right_index
        ]
        model_input[converter.eye_happy_wink_right_index] = (
            model_input[converter.eye_happy_wink_left_index] / 2
        )
        model_input[converter.eye_happy_wink_left_index] = (
            model_input[converter.eye_happy_wink_left_index] / 2
        )
        model_input[converter.mouth_aaa_index] = min(
            model_input[converter.mouth_aaa_index]
            + model_input[converter.mouth_ooo_index] / 2
            + model_input[converter.mouth_iii_index] / 2
            + model_input[converter.mouth_uuu_index] / 2,
            1,
        )
        model_input[converter.mouth_ooo_index] = 0
        model_input[converter.mouth_iii_index] = 0
        model_input[converter.mouth_uuu_index] = 0

    scales, quantized = _get_simplify_config(simplify)
    model_input[quantized] = (
        np.round(model_input[quantized] * scales[quantized]) / scales[quantized]
    )

    input_pose = np.zeros((1, 45), dtype=np.float32)
    if args.eyebrow:
        input_pose[0, :12] = model_input[:12]
    input_pose[0, 12:] = model_input[12:45]
    return input_pose
