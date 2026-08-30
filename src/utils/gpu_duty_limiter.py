import math
import time


class GpuDutyCycleLimiter:
    """Limit the sustained duty cycle of synchronous inference calls.

    The limiter schedules the earliest start time for the next inference. Work
    performed elsewhere in the pipeline naturally counts as idle time, so an
    existing FPS/back-pressure delay is not paid twice.
    """

    def __init__(self, limit_percent, clock=None, sleeper=None):
        limit_percent = float(limit_percent)
        if not math.isfinite(limit_percent) or limit_percent <= 0 or limit_percent > 100:
            raise ValueError("limit_percent must be greater than 0 and at most 100")

        self.limit_percent = limit_percent
        self._limit_fraction = limit_percent / 100.0
        self._clock = clock or time.perf_counter
        self._sleeper = sleeper or time.sleep
        self._next_start_time = None

    @property
    def enabled(self):
        return self._limit_fraction < 1.0

    def wait(self):
        """Wait until another inference can start and return seconds waited."""
        if not self.enabled or self._next_start_time is None:
            return 0.0

        wait_started_at = self._clock()
        remaining = self._next_start_time - wait_started_at
        while remaining > 0:
            self._sleeper(remaining)
            remaining = self._next_start_time - self._clock()
        return max(0.0, self._clock() - wait_started_at)

    def record_inference(self, started_at, finished_at=None):
        """Record one synchronous inference and return its scheduled cooldown."""
        if not self.enabled:
            self._next_start_time = None
            return 0.0

        if finished_at is None:
            finished_at = self._clock()
        active_seconds = max(0.0, finished_at - started_at)
        cooldown_seconds = active_seconds * (1.0 / self._limit_fraction - 1.0)
        self._next_start_time = finished_at + cooldown_seconds
        return cooldown_seconds
