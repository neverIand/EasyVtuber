"""Capture a launcher page for repeatable visual regression checks.

The requested width and height are device-independent pixels (DIP), matching
the sizes used by launcher2.py. By default the currently selected character
is drawn in the persistent preview pane so both halves of the layout are
covered by the screenshot.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import cv2
import numpy as np
from PIL import ImageGrab
import wx

import launcher2
from src.utils.preview_ipc import PreviewFrameFormatter


PAGE_INDEX = {
    'basic': 0,
    'performance': 1,
    'advanced': 2,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Capture a launcher page at a repeatable DIP size.',
    )
    parser.add_argument(
        '--page',
        choices=tuple(PAGE_INDEX),
        default='basic',
        help='Settings page to show (default: basic).',
    )
    parser.add_argument(
        '--width',
        type=int,
        default=1180,
        help='Launcher client width in DIP (default: 1180).',
    )
    parser.add_argument(
        '--height',
        type=int,
        default=700,
        help='Launcher client height in DIP (default: 700).',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=PROJECT_ROOT / 'benchmark_results' / 'launcher-ui.png',
        help='PNG output path.',
    )
    parser.add_argument(
        '--sample-image',
        type=Path,
        help='Optional character PNG to draw in the preview pane.',
    )
    parser.add_argument(
        '--empty-preview',
        action='store_true',
        help='Leave the preview pane in its idle state.',
    )
    return parser.parse_args()


def selected_character_path():
    model_name = str(launcher2.args.get('model_select', ''))
    if 'tha4_student_' in model_name:
        student_name = model_name.replace('tha4_student_', '', 1)
        candidate = launcher2.studentModelCharacterMap.get(student_name)
        if candidate:
            path = PROJECT_ROOT / candidate
            if path.is_file():
                return path
    elif model_name == 'tha4_student':
        path = PROJECT_ROOT / 'data' / 'models' / 'tha4_student' / 'character.png'
        if path.is_file():
            return path

    character_name = str(launcher2.args.get('character', ''))
    path = PROJECT_ROOT / 'data' / 'images' / f'{character_name}.png'
    if path.is_file():
        return path

    return next(
        (path for path in sorted((PROJECT_ROOT / 'data' / 'images').glob('*.png'))),
        None,
    )


def load_preview_frame(path):
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f'Could not decode preview image: {path}')
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f'Preview image must have three or four channels: {path}')

    source_format = 'BGRA' if image.shape[2] == 4 else 'BGR'
    return PreviewFrameFormatter().format(image, source_format).copy()


def let_wx_paint(frame):
    frame.Raise()
    for _ in range(10):
        frame.Refresh()
        frame.Update()
        wx.YieldIfNeeded()
        wx.MilliSleep(50)


def capture_window(frame, output_path):
    width, height = frame.GetClientSize()
    try:
        origin = frame.ClientToScreen(wx.Point(0, 0))
        screenshot = ImageGrab.grab(
            bbox=(origin.x, origin.y, origin.x + width, origin.y + height),
            include_layered_windows=True,
        )
    except (OSError, ValueError):
        bitmap = wx.Bitmap(width, height, 32)
        destination = wx.MemoryDC(bitmap)
        source = wx.ClientDC(frame)
        try:
            if not destination.Blit(0, 0, width, height, source, 0, 0):
                raise RuntimeError('wxWidgets could not copy the launcher window')
        finally:
            destination.SelectObject(wx.NullBitmap)
        screenshot = bitmap.ConvertToImage()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(screenshot, wx.Image):
        if not screenshot.SaveFile(str(output_path), wx.BITMAP_TYPE_PNG):
            raise RuntimeError(f'Could not save screenshot: {output_path}')
    else:
        screenshot.save(output_path, format='PNG')
    return width, height


def main():
    cli = parse_args()
    if cli.width <= 0 or cli.height <= 0:
        raise SystemExit('--width and --height must be positive')

    app = wx.App(False)
    frame = launcher2.MainFrame(None)
    try:
        frame.SetClientSize(frame.FromDIP(wx.Size(cli.width, cli.height)))
        frame.panel.notebook.SetSelection(PAGE_INDEX[cli.page])

        if not cli.empty_preview:
            image_path = (
                cli.sample_image.resolve()
                if cli.sample_image is not None
                else selected_character_path()
            )
            if image_path is not None:
                frame.panel.previewCanvas.SetFrame(load_preview_frame(image_path))
                frame.panel._set_preview_status(
                    f'界面回归示例 · {image_path.name}'
                )

        frame.panel._layout_options()
        frame.Centre()
        frame.Show()
        let_wx_paint(frame)

        output_path = cli.output.resolve()
        pixel_width, pixel_height = capture_window(frame, output_path)
        print(
            f'Captured {cli.page} page at {cli.width}x{cli.height} DIP '
            f'({pixel_width}x{pixel_height} px): {output_path}'
        )
    finally:
        frame.Destroy()
        wx.YieldIfNeeded()


if __name__ == '__main__':
    main()
