"""Fast display-adapter detection without spawning deprecated WMIC."""

import ctypes
import sys

from ctypes import wintypes
from typing import Iterable, List


class _DisplayDeviceW(ctypes.Structure):
    _fields_ = [
        ('cb', wintypes.DWORD),
        ('DeviceName', wintypes.WCHAR * 32),
        ('DeviceString', wintypes.WCHAR * 128),
        ('StateFlags', wintypes.DWORD),
        ('DeviceID', wintypes.WCHAR * 128),
        ('DeviceKey', wintypes.WCHAR * 128),
    ]


def _adapter_names_from_enum(enum_display_devices) -> List[str]:
    names = []
    adapter_index = 0
    while True:
        device = _DisplayDeviceW()
        device.cb = ctypes.sizeof(device)
        if not enum_display_devices(
            None,
            adapter_index,
            ctypes.byref(device),
            0,
        ):
            break
        if device.DeviceString and device.DeviceString not in names:
            names.append(device.DeviceString)
        adapter_index += 1
    return names


def display_adapter_names() -> List[str]:
    """Return installed Windows display adapter names."""
    if sys.platform != 'win32':
        return []
    try:
        user32 = ctypes.WinDLL('user32', use_last_error=True)
        enum_display_devices = user32.EnumDisplayDevicesW
        enum_display_devices.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            ctypes.POINTER(_DisplayDeviceW),
            wintypes.DWORD,
        ]
        enum_display_devices.restype = wintypes.BOOL
        return _adapter_names_from_enum(enum_display_devices)
    except (AttributeError, OSError):
        return []


def contains_nvidia_adapter(adapter_names: Iterable[str]) -> bool:
    return any('NVIDIA' in name.upper() for name in adapter_names)


def has_nvidia_gpu() -> bool:
    return contains_nvidia_adapter(display_adapter_names())
