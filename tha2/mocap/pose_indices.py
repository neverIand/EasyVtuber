"""Lightweight pose-index schema shared by runtime input code.

The indices are the stable layout produced by
``tha2.poser.modes.mode_20.get_pose_parameters``.  Keeping the layout here
prevents input-only code from importing PyTorch and the legacy THA networks
just to look up constants.
"""


class PoseIndices20:
    pose_size = 45

    eyebrow_troubled_left_index = 0
    eyebrow_troubled_right_index = 1
    eyebrow_angry_left_index = 2
    eyebrow_angry_right_index = 3
    eyebrow_lowered_left_index = 4
    eyebrow_lowered_right_index = 5
    eyebrow_raised_left_index = 6
    eyebrow_raised_right_index = 7
    eyebrow_happy_left_index = 8
    eyebrow_happy_right_index = 9
    eyebrow_serious_left_index = 10
    eyebrow_serious_right_index = 11

    eye_wink_left_index = 12
    eye_wink_right_index = 13
    eye_happy_wink_left_index = 14
    eye_happy_wink_right_index = 15
    eye_surprised_left_index = 16
    eye_surprised_right_index = 17
    eye_relaxed_left_index = 18
    eye_relaxed_right_index = 19
    eye_unimpressed_left_index = 20
    eye_unimpressed_right_index = 21
    eye_raised_lower_eyelid_left_index = 22
    eye_raised_lower_eyelid_right_index = 23

    iris_small_left_index = 24
    iris_small_right_index = 25

    mouth_aaa_index = 26
    mouth_iii_index = 27
    mouth_uuu_index = 28
    mouth_eee_index = 29
    mouth_ooo_index = 30
    mouth_delta_index = 31
    mouth_lowered_corner_left_index = 32
    mouth_lowered_corner_right_index = 33
    mouth_raised_corner_left_index = 34
    mouth_raised_corner_right_index = 35
    mouth_smirk_index = 36

    iris_rotation_x_index = 37
    iris_rotation_y_index = 38
    head_x_index = 39
    head_y_index = 40
    neck_z_index = 41


POSE_INDICES_20 = PoseIndices20()

