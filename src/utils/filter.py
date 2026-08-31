"""Vectorized NumPy implementation of the One Euro Filter."""

import math
import time
from typing import Optional

import numpy as np


class OneEuroFilterNumpy:
    """Apply independent One Euro Filters to every value in a NumPy array.

    All channels share the same timestamp and sampling frequency, so their
    scalar filter state can be represented by NumPy arrays without changing
    the algorithm used by the original ``OneEuroFilter`` package.
    """

    def __init__(
        self,
        freq: float,
        mincutoff: float = 1.0,
        beta: float = 0.0,
        dcutoff: float = 1.0,
    ) -> None:
        self._validate_positive('freq', freq)
        self._validate_positive('mincutoff', mincutoff)
        self._validate_positive('dcutoff', dcutoff)

        self.freq = float(freq)
        self.mincutoff = float(mincutoff)
        self.beta = float(beta)
        self.dcutoff = float(dcutoff)
        self._shape = None
        self._x_filtered = None
        self._dx_filtered = None
        self._has_x_filtered = False
        self._x_initialized = False
        self._dx_initialized = False
        self._lasttime = None

    @staticmethod
    def _validate_positive(name: str, value: float) -> None:
        if value <= 0:
            raise ValueError('{} should be >0'.format(name))

    def _alpha(self, cutoff):
        te = 1.0 / self.freq
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def __call__(
        self,
        x: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> np.ndarray:
        values = np.asarray(x, dtype=np.float64)
        if self._shape is None:
            self._shape = values.shape
            self._x_filtered = np.empty(self._shape, dtype=np.float64)
            self._dx_filtered = np.empty(self._shape, dtype=np.float64)
        elif values.shape != self._shape:
            raise ValueError(
                "Input shape {} doesn't match initialized shape {}".format(
                    values.shape,
                    self._shape,
                )
            )

        if timestamp is None:
            timestamp = time.perf_counter()
        if self._lasttime and timestamp and timestamp > self._lasttime:
            self.freq = 1.0 / (timestamp - self._lasttime)
        self._lasttime = timestamp

        if self._has_x_filtered:
            derivative = (values - self._x_filtered) * self.freq
        else:
            derivative = np.zeros_like(values)

        if self._dx_initialized:
            derivative_alpha = self._alpha(self.dcutoff)
            self._dx_filtered[:] = (
                derivative_alpha * derivative
                + (1.0 - derivative_alpha) * self._dx_filtered
            )
        else:
            self._dx_filtered[:] = derivative
            self._dx_initialized = True

        cutoff = self.mincutoff + self.beta * np.abs(self._dx_filtered)
        if self._x_initialized:
            value_alpha = self._alpha(cutoff)
            self._x_filtered[:] = (
                value_alpha * values
                + (1.0 - value_alpha) * self._x_filtered
            )
        else:
            self._x_filtered[:] = values
            self._x_initialized = True
            self._has_x_filtered = True

        return self._x_filtered.copy()

    def filter(
        self,
        x: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> np.ndarray:
        """Filter an array; alias for :meth:`__call__`."""
        return self.__call__(x, timestamp)

    def reset(self) -> None:
        # Match the scalar package: LowPassFilter.reset() clears its current
        # sample marker but retains the last filtered value used for dx.
        if self._shape is not None:
            self._x_initialized = False
            self._dx_initialized = False
            self._lasttime = None

    def setFrequency(self, freq: float) -> None:
        self._validate_positive('freq', freq)
        self.freq = float(freq)

    def setMinCutoff(self, mincutoff: float) -> None:
        self._validate_positive('mincutoff', mincutoff)
        self.mincutoff = float(mincutoff)

    def setBeta(self, beta: float) -> None:
        self.beta = float(beta)

    def setDerivateCutoff(self, dcutoff: float) -> None:
        self._validate_positive('dcutoff', dcutoff)
        self.dcutoff = float(dcutoff)

    def setParameters(
        self,
        freq: float,
        mincutoff: float = 1.0,
        beta: float = 0.0,
        dcutoff: float = 1.0,
    ) -> None:
        self.setFrequency(freq)
        self.setMinCutoff(mincutoff)
        self.setBeta(beta)
        self.setDerivateCutoff(dcutoff)
