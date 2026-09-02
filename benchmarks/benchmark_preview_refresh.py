"""Measure shared-memory publishing and wx preview consumption rates.

This is a CPU-only launcher diagnostic: it feeds synthetic RGBA frames into
the same shared memory and wx timer used by the real child process.  It does
not load a character model or touch the GPU.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import threading
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np
import wx

import launcher2
from src.utils.preview_ipc import PreviewPublishPacer, PreviewSharedBuffer
from src.utils.timer_wait import wait_until


def measured_rate(timestamps):
    if len(timestamps) < 2:
        return 0.0
    elapsed = timestamps[-1] - timestamps[0]
    return (len(timestamps) - 1) / elapsed if elapsed > 0 else 0.0


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Feed synthetic frames through the launcher preview without '
            'starting a model or GPU backend.'
        ),
    )
    parser.add_argument('--source-fps', type=float, default=30.0)
    parser.add_argument(
        '--publish-cap',
        type=float,
        default=0.0,
        help='Shared-memory publication cap; use 0 to disable (default: 0).',
    )
    parser.add_argument('--seconds', type=float, default=4.0)
    return parser.parse_args()


def main():
    cli = parse_args()
    if cli.source_fps <= 0 or cli.seconds <= 0:
        raise SystemExit('--source-fps and --seconds must be positive')
    if cli.publish_cap < 0:
        raise SystemExit('--publish-cap must be non-negative')

    app = wx.App(False)
    frame = launcher2.MainFrame(None)
    panel = frame.panel
    consumed_times = []
    source_times = []
    published_times = []
    publisher_errors = []
    stop_event = threading.Event()

    original_set_frame = panel.previewCanvas.SetFrame

    def counted_set_frame(preview_frame):
        consumed_times.append(time.perf_counter())
        original_set_frame(preview_frame)

    panel.previewCanvas.SetFrame = counted_set_frame
    preview_name = panel.StartPreview()
    if not preview_name:
        frame.Destroy()
        raise SystemExit('Could not create launcher preview shared memory')

    writer = PreviewSharedBuffer.attach(preview_name)
    source = np.zeros((writer.height, writer.width, 4), dtype=np.uint8)
    source[:, :, 3] = 255
    publish_pacer = (
        PreviewPublishPacer(cli.publish_cap)
        if cli.publish_cap > 0
        else None
    )

    def publish_frames():
        frame_count = max(1, round(cli.source_fps * cli.seconds))
        deadline = time.perf_counter() + 0.2
        interval = 1.0 / cli.source_fps
        try:
            for sequence in range(frame_count):
                if stop_event.is_set():
                    break
                wait_until(deadline)
                now = time.perf_counter()
                source_times.append(now)
                source[0, 0, 0] = sequence & 0xFF
                if publish_pacer is None or publish_pacer.is_due(now):
                    writer.publish_rgba(source)
                    published_times.append(time.perf_counter())
                deadline += interval
        except Exception as error:
            publisher_errors.append(error)

    publisher = threading.Thread(target=publish_frames, daemon=True)

    def finish():
        stop_event.set()
        publisher.join(timeout=1.0)
        writer.close()
        panel.StopPreview(message='刷新率诊断完成', clear=True)
        frame.Destroy()
        app.ExitMainLoop()

    frame.Show()
    frame.Raise()
    publisher.start()
    wx.CallLater(round((cli.seconds + 0.8) * 1000), finish)
    app.MainLoop()
    publisher.join(timeout=1.0)

    if publisher_errors:
        raise publisher_errors[0]

    source_rate = measured_rate(source_times)
    published_rate = measured_rate(published_times)
    consumed_rate = measured_rate(consumed_times)
    print(
        f'synthetic source: {len(source_times)} frames, '
        f'{source_rate:.2f} FPS'
    )
    print(
        f'shared-memory published: {len(published_times)} frames, '
        f'{published_rate:.2f} FPS'
    )
    print(
        f'wx preview consumed: {len(consumed_times)} frames, '
        f'{consumed_rate:.2f} FPS'
    )
    if published_times:
        print(
            f'consumed/published: '
            f'{100.0 * len(consumed_times) / len(published_times):.1f}%'
        )


if __name__ == '__main__':
    main()
