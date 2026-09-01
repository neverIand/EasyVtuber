"""Low-CPU deadline waits for the Windows runtime."""

import atexit
import ctypes
import math
import sys
import threading
import time

from ctypes import wintypes


_CREATE_WAITABLE_TIMER_HIGH_RESOLUTION = 0x00000002
_SYNCHRONIZE = 0x00100000
_TIMER_MODIFY_STATE = 0x0002
_WAIT_OBJECT_0 = 0x00000000
_INFINITE = 0xFFFFFFFF


class _HighResolutionWaitableTimer:
    def __init__(self, kernel32, handle):
        self._kernel32 = kernel32
        self._handle = handle

    def wait(self, seconds: float) -> bool:
        """Wait for a relative duration, returning False on a Win32 error."""
        due_time = ctypes.c_longlong(
            -max(1, int(seconds * 10_000_000))
        )
        if not self._kernel32.SetWaitableTimerEx(
            self._handle,
            ctypes.byref(due_time),
            0,
            None,
            None,
            None,
            0,
        ):
            return False
        return (
            self._kernel32.WaitForSingleObject(self._handle, _INFINITE)
            == _WAIT_OBJECT_0
        )

    def close(self) -> None:
        if self._handle:
            self._kernel32.CloseHandle(self._handle)
            self._handle = None


def _create_high_resolution_timer():
    if sys.platform != 'win32':
        return None

    try:
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel32.CreateWaitableTimerExW.argtypes = [
            ctypes.c_void_p,
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
        ]
        kernel32.CreateWaitableTimerExW.restype = wintypes.HANDLE
        kernel32.SetWaitableTimerEx.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ctypes.c_longlong),
            wintypes.LONG,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            wintypes.ULONG,
        ]
        kernel32.SetWaitableTimerEx.restype = wintypes.BOOL
        kernel32.WaitForSingleObject.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
        ]
        kernel32.WaitForSingleObject.restype = wintypes.DWORD
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL

        handle = kernel32.CreateWaitableTimerExW(
            None,
            None,
            _CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
            _SYNCHRONIZE | _TIMER_MODIFY_STATE,
        )
        if not handle:
            return None
        return _HighResolutionWaitableTimer(kernel32, handle)
    except (AttributeError, OSError):
        return None


def _request_timer_resolution() -> None:
    """Request 0.5 ms scheduling resolution and release it on process exit."""
    if sys.platform != 'win32':
        return
    try:
        ntdll = ctypes.WinDLL('ntdll')
        ntdll.NtSetTimerResolution.argtypes = [
            ctypes.c_ulong,
            wintypes.BOOL,
            ctypes.POINTER(ctypes.c_ulong),
        ]
        ntdll.NtSetTimerResolution.restype = ctypes.c_long
        desired = ctypes.c_ulong(5000)
        current = ctypes.c_ulong()
        status = ntdll.NtSetTimerResolution(
            desired,
            True,
            ctypes.byref(current),
        )
        if status < 0:
            return

        def release_resolution():
            ntdll.NtSetTimerResolution(
                desired,
                False,
                ctypes.byref(current),
            )

        atexit.register(release_resolution)
    except (AttributeError, OSError):
        pass


_waitable_timer = _create_high_resolution_timer()
_waitable_timer_lock = threading.Lock()
_request_timer_resolution()
if _waitable_timer is not None:
    atexit.register(_waitable_timer.close)


def wait_backend_name() -> str:
    """Return the selected implementation for diagnostics and tests."""
    if _waitable_timer is not None:
        return 'high-resolution-waitable-timer'
    return 'sleep-spin-fallback'


def wait_until(
    target_time: float,
    spin_threshold: float = 0.0005,
    sleep_min: float = 0.001,
):
    """Wait until ``perf_counter()`` reaches an absolute deadline.

    Modern Windows uses a reusable high-resolution waitable timer, avoiding
    the old per-frame 0.5 ms busy-spin window. The previous sleep/spin path is
    retained as a compatibility fallback for unsupported systems.
    """
    if not isinstance(target_time, (int, float)):
        return
    target_time = float(target_time)
    if not math.isfinite(target_time):
        return

    if _waitable_timer is not None:
        while True:
            remaining = target_time - time.perf_counter()
            if remaining <= 0:
                return
            with _waitable_timer_lock:
                remaining = target_time - time.perf_counter()
                if remaining <= 0:
                    return
                if not _waitable_timer.wait(remaining):
                    break

    while True:
        remaining = target_time - time.perf_counter()
        if remaining <= 0:
            return
        if remaining <= spin_threshold:
            break

        sleep_time = remaining - spin_threshold
        if sleep_time >= sleep_min:
            time.sleep(sleep_time)
        else:
            break

    perf_counter = time.perf_counter
    while perf_counter() < target_time:
        pass
