import cv2
import numpy as np


def prepare_mediapipe_rgb(frame: np.ndarray, output=None) -> np.ndarray:
    """Convert a BGR camera frame into a reusable, read-only RGB buffer."""
    if output is None or output.shape != frame.shape or output.dtype != frame.dtype:
        output = np.empty(frame.shape, dtype=frame.dtype)
    else:
        output.flags.writeable = True

    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, dst=output)
    output.flags.writeable = False
    return output
