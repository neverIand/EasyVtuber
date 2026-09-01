"""Reusable staging buffer for the legacy PySpout int32 binding."""

from typing import Sequence

import numpy as np


def create_spout_staging_buffer(shape: Sequence[int]) -> np.ndarray:
    shape = tuple(shape)
    if len(shape) != 3 or shape[2] != 4:
        raise ValueError('Spout frames must use an HWC shape with 4 channels')
    return np.empty(shape, dtype=np.int32, order='C')


def stage_spout_frame(
        source: np.ndarray,
        destination: np.ndarray,
) -> np.ndarray:
    """Copy one uint8 RGBA frame into a reusable contiguous int32 array.

    The bundled PySpout extension declares ``py::array_t<int>``. Passing the
    normal uint8 output makes pybind11 allocate and cast a temporary array for
    every frame. Supplying this pre-cast buffer avoids that allocation while
    preserving the exact low byte consumed by the existing C++ binding.
    """
    if source.dtype != np.uint8:
        raise TypeError('Spout source must have dtype uint8')
    if destination.dtype != np.int32:
        raise TypeError('Spout destination must have dtype int32')
    if source.shape != destination.shape:
        raise ValueError('Spout source and destination shapes must match')
    if not destination.flags.c_contiguous:
        raise ValueError('Spout destination must be C-contiguous')

    np.copyto(destination, source, casting='safe')
    return destination
