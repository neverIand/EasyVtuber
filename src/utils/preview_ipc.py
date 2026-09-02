"""Non-blocking shared-memory transport for the launcher's latest-frame preview."""

from __future__ import annotations

import struct
from multiprocessing import shared_memory

import numpy as np


PREVIEW_WIDTH = 512
PREVIEW_HEIGHT = 512
PREVIEW_CHANNELS = 4
PREVIEW_FPS = 30
# wx.Timer uses the coarse WM_TIMER clock on Windows. Polling at the same
# 33 ms cadence can be rounded up to roughly 46.9 ms (about 21 FPS). A cheap
# header poll at 15 ms keeps the consumer ahead of the capped 30 FPS writer;
# bitmap creation and repaint still happen only when a new frame is present.
PREVIEW_UI_POLL_MS = 15

_MAGIC = b'EVP1'
_HEADER = struct.Struct('<4sIIII')
_HEADER_SIZE = 64
_SEQUENCE_OFFSET = 12
_ACTIVE_INDEX_OFFSET = 16


class PreviewPublishPacer:
    """Cap preview publication without losing phase against the source."""

    def __init__(self, fps=PREVIEW_FPS):
        if fps <= 0:
            raise ValueError('Preview FPS must be positive')
        self.interval = 1.0 / fps
        self._next_time = None

    def is_due(self, now):
        now = float(now)
        if self._next_time is None:
            self._next_time = now + self.interval
            return True
        if now + 1e-9 < self._next_time:
            return False

        # Advance the existing deadline to retain phase. Reset only after a
        # whole missed period so a long stall cannot trigger a catch-up burst.
        next_time = self._next_time + self.interval
        self._next_time = (
            now + self.interval
            if next_time <= now + 1e-9
            else next_time
        )
        return True


class PreviewSharedBuffer:
    """Latest-frame RGBA exchange where the reader never blocks the writer."""

    def __init__(self, memory, owner=False):
        self._memory = memory
        self._owner = owner
        magic, width, height, sequence, active_index = _HEADER.unpack_from(
            memory.buf,
        )
        if magic != _MAGIC:
            raise ValueError('Invalid preview shared-memory header')
        if width <= 0 or height <= 0 or active_index not in (0, 1):
            raise ValueError('Invalid preview shared-memory dimensions')

        expected_size = _HEADER_SIZE + 2 * width * height * PREVIEW_CHANNELS
        if memory.size < expected_size:
            raise ValueError('Preview shared memory is smaller than its header')

        self.width = width
        self.height = height
        self._frames = np.ndarray(
            (2, height, width, PREVIEW_CHANNELS),
            dtype=np.uint8,
            buffer=memory.buf,
            offset=_HEADER_SIZE,
        )
        self._read_frame = np.empty(
            (height, width, PREVIEW_CHANNELS),
            dtype=np.uint8,
        )
        self._last_sequence = sequence

    @classmethod
    def create(cls, width=PREVIEW_WIDTH, height=PREVIEW_HEIGHT):
        if width <= 0 or height <= 0:
            raise ValueError('Preview dimensions must be positive')
        size = _HEADER_SIZE + 2 * width * height * PREVIEW_CHANNELS
        memory = shared_memory.SharedMemory(create=True, size=size)
        try:
            _HEADER.pack_into(memory.buf, 0, _MAGIC, width, height, 0, 0)
            instance = cls(memory, owner=True)
            instance._frames.fill(0)
            return instance
        except Exception:
            memory.close()
            memory.unlink()
            raise

    @classmethod
    def attach(cls, name):
        memory = shared_memory.SharedMemory(name=name)
        try:
            return cls(memory, owner=False)
        except Exception:
            memory.close()
            raise

    @property
    def name(self):
        return self._memory.name

    def publish_rgba(self, frame):
        if frame.dtype != np.uint8:
            raise TypeError('Preview frame must use uint8 pixels')
        if frame.shape != (self.height, self.width, PREVIEW_CHANNELS):
            raise ValueError('Preview frame shape does not match shared memory')

        sequence = struct.unpack_from(
            '<I', self._memory.buf, _SEQUENCE_OFFSET,
        )[0]
        active_index = struct.unpack_from(
            '<I', self._memory.buf, _ACTIVE_INDEX_OFFSET,
        )[0]
        next_index = 1 - active_index

        # Publish the index and then the sequence only after the inactive frame
        # is complete. Readers reject a copy if the sequence changes around it.
        np.copyto(self._frames[next_index], frame)
        struct.pack_into(
            '<I', self._memory.buf, _ACTIVE_INDEX_OFFSET, next_index,
        )
        struct.pack_into(
            '<I',
            self._memory.buf,
            _SEQUENCE_OFFSET,
            (sequence + 1) & 0xFFFFFFFF,
        )

    def read_latest(self):
        sequence_before = struct.unpack_from(
            '<I', self._memory.buf, _SEQUENCE_OFFSET,
        )[0]
        if sequence_before == self._last_sequence:
            return None
        active_index = struct.unpack_from(
            '<I', self._memory.buf, _ACTIVE_INDEX_OFFSET,
        )[0]
        if active_index not in (0, 1):
            return None

        np.copyto(self._read_frame, self._frames[active_index])
        sequence_after = struct.unpack_from(
            '<I', self._memory.buf, _SEQUENCE_OFFSET,
        )[0]
        if sequence_before != sequence_after:
            return None
        self._last_sequence = sequence_after
        return self._read_frame

    def close(self, unlink=None):
        if self._memory is None:
            return
        memory = self._memory
        owner = self._owner
        self._frames = None
        self._read_frame = None
        self._memory = None
        memory.close()
        should_unlink = owner if unlink is None else unlink
        if should_unlink:
            try:
                memory.unlink()
            except FileNotFoundError:
                pass


class PreviewFrameFormatter:
    """Convert an output frame into a centered, fixed-size RGBA preview."""

    _FORMATS = {
        'BGR': (3, 'COLOR_BGR2RGBA'),
        'RGB': (3, 'COLOR_RGB2RGBA'),
        'BGRA': (4, 'COLOR_BGRA2RGBA'),
        'RGBA': (4, None),
    }

    def __init__(self, width=PREVIEW_WIDTH, height=PREVIEW_HEIGHT):
        if width <= 0 or height <= 0:
            raise ValueError('Preview dimensions must be positive')
        self.width = width
        self.height = height
        self._canvas = np.zeros(
            (height, width, PREVIEW_CHANNELS),
            dtype=np.uint8,
        )
        self._resized = None
        self._resized_shape = None

    def format(self, frame, source_format):
        import cv2

        if frame.dtype != np.uint8 or frame.ndim != 3:
            raise TypeError('Preview source must be an HWC uint8 array')
        try:
            channels, conversion_name = self._FORMATS[source_format]
        except KeyError as error:
            raise ValueError(f'Unsupported preview format: {source_format}') from error
        if frame.shape[2] != channels:
            raise ValueError('Preview source channels do not match its format')

        source_height, source_width = frame.shape[:2]
        scale = min(
            self.width / source_width,
            self.height / source_height,
        )
        target_width = max(1, min(self.width, round(source_width * scale)))
        target_height = max(1, min(self.height, round(source_height * scale)))
        target_shape = (target_height, target_width, channels)

        if (source_height, source_width) == (target_height, target_width):
            resized = frame
        else:
            if self._resized_shape != target_shape:
                self._resized = np.empty(target_shape, dtype=np.uint8)
                self._resized_shape = target_shape
            interpolation = (
                cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            )
            cv2.resize(
                frame,
                (target_width, target_height),
                dst=self._resized,
                interpolation=interpolation,
            )
            resized = self._resized

        self._canvas.fill(0)
        left = (self.width - target_width) // 2
        top = (self.height - target_height) // 2
        destination = self._canvas[
            top:top + target_height,
            left:left + target_width,
        ]
        if conversion_name is None:
            np.copyto(destination, resized)
        else:
            cv2.cvtColor(
                resized,
                getattr(cv2, conversion_name),
                dst=destination,
            )
        return self._canvas
