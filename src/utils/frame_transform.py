import math
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np


def build_output_transform(
    position: Sequence[float],
    frame_shape: Tuple[int, ...],
    extend_movement: bool,
    bongo: bool,
) -> Optional[np.ndarray]:
    """Build the legacy output transform, returning None for an identity transform."""
    scale = 1.0
    angle = 0.0
    dx = 0.0
    dy = 0.0

    if extend_movement:
        scale = position[2] * math.sqrt(extend_movement) + 1
        angle = -position[0] * 10 * extend_movement
        dx = position[0] * 400 * scale * extend_movement
        dy = -position[1] * 600 * scale * extend_movement
    if bongo:
        angle -= 5.0

    if scale == 1.0 and angle == 0.0 and dx == 0.0 and dy == 0.0:
        return None

    height, width = frame_shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
    matrix[0, 2] += dx
    matrix[1, 2] += dy
    return matrix


def apply_output_transform(
    frame: np.ndarray,
    transform: Optional[np.ndarray],
    copy_identity: bool = False,
) -> np.ndarray:
    """Apply an output transform without running warpAffine for an identity matrix."""
    if transform is None:
        return frame.copy() if copy_identity else frame
    return cv2.warpAffine(frame, transform, (frame.shape[1], frame.shape[0]))
