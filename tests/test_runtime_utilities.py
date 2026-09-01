import ctypes
import math
import time
import unittest
from unittest.mock import patch

from src.utils.fps import FPS
from src.utils.gpu_detect import (
    _DisplayDeviceW,
    _adapter_names_from_enum,
    contains_nvidia_adapter,
)
from src.utils.timer_wait import wait_backend_name, wait_until


class RuntimeUtilityTests(unittest.TestCase):
    def test_fps_counts_intervals_between_timestamps(self):
        fps = FPS()
        with patch(
            'src.utils.fps.time.perf_counter',
            side_effect=(10.0, 10.5, 11.0),
        ):
            self.assertEqual(fps(), 0.0)
            self.assertAlmostEqual(fps(), 2.0)
            self.assertAlmostEqual(fps(), 2.0)

    def test_adapter_enumeration_and_case_insensitive_nvidia_match(self):
        adapter_names = ('Intel(R) Graphics', 'Nvidia GeForce Test GPU')

        def fake_enum(_device_name, index, device_pointer, _flags):
            if index >= len(adapter_names):
                return False
            device = ctypes.cast(
                device_pointer,
                ctypes.POINTER(_DisplayDeviceW),
            ).contents
            device.DeviceString = adapter_names[index]
            return True

        self.assertEqual(
            _adapter_names_from_enum(fake_enum),
            list(adapter_names),
        )
        self.assertTrue(contains_nvidia_adapter(adapter_names))
        self.assertFalse(contains_nvidia_adapter(('Intel', 'AMD')))

    def test_wait_until_reaches_deadline_and_ignores_invalid_values(self):
        target = time.perf_counter() + 0.002
        wait_until(target)
        self.assertGreaterEqual(time.perf_counter(), target)

        for invalid in (None, 'later', math.inf, math.nan):
            wait_until(invalid)

        self.assertIn(
            wait_backend_name(),
            ('high-resolution-waitable-timer', 'sleep-spin-fallback'),
        )


if __name__ == '__main__':
    unittest.main()
