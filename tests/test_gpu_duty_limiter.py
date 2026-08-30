import math
import unittest

from src.utils.gpu_duty_limiter import GpuDutyCycleLimiter


class FakeTime:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def perf_counter(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class GpuDutyCycleLimiterTests(unittest.TestCase):
    def test_90_percent_adds_ten_ms_after_ninety_ms_inference(self):
        fake_time = FakeTime()
        limiter = GpuDutyCycleLimiter(
            90,
            clock=fake_time.perf_counter,
            sleeper=fake_time.sleep,
        )

        started_at = fake_time.perf_counter()
        fake_time.now = 0.09
        cooldown = limiter.record_inference(started_at)
        waited = limiter.wait()

        self.assertAlmostEqual(cooldown, 0.01)
        self.assertAlmostEqual(waited, 0.01)
        self.assertAlmostEqual(fake_time.now, 0.10)

    def test_pipeline_idle_time_avoids_duplicate_wait(self):
        fake_time = FakeTime()
        limiter = GpuDutyCycleLimiter(
            80,
            clock=fake_time.perf_counter,
            sleeper=fake_time.sleep,
        )

        limiter.record_inference(0.0, finished_at=0.08)
        fake_time.now = 0.11

        self.assertEqual(limiter.wait(), 0.0)
        self.assertEqual(fake_time.sleeps, [])

    def test_100_percent_disables_throttling(self):
        fake_time = FakeTime()
        limiter = GpuDutyCycleLimiter(
            100,
            clock=fake_time.perf_counter,
            sleeper=fake_time.sleep,
        )

        self.assertFalse(limiter.enabled)
        self.assertEqual(limiter.record_inference(0.0, finished_at=1.0), 0.0)
        self.assertEqual(limiter.wait(), 0.0)

    def test_invalid_limits_are_rejected(self):
        for value in (0, -1, 100.1, math.inf, math.nan):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    GpuDutyCycleLimiter(value)


if __name__ == '__main__':
    unittest.main()
