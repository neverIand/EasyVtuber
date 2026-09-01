import sys
import unittest
from contextlib import contextmanager
from types import SimpleNamespace

import cv2
import numpy as np


original_argv = sys.argv
try:
    sys.argv = [sys.argv[0]]
    from src.args import args
    from src.model_infer_client import ModelClientProcess
    from src.utils.frame_transform import apply_output_transform, build_output_transform
finally:
    sys.argv = original_argv


class _Guard:
    def __init__(self):
        self.entries = 0

    @contextmanager
    def lock(self):
        self.entries += 1
        yield


class ModelPostprocessTests(unittest.TestCase):
    _ARG_NAMES = (
        'output_debug',
        'output_virtual_cam',
        'output_spout2',
        'alpha_split',
        'extend_movement',
        'bongo',
        'max_ram_cache_len',
        'max_gpu_cache_len',
    )

    def setUp(self):
        self.original_args = {
            name: getattr(args, name)
            for name in self._ARG_NAMES
        }
        self.process = object.__new__(ModelClientProcess)
        self.process.pipeline_fps_number = SimpleNamespace(value=29.98765)
        self.process.input_fps = SimpleNamespace(value=59.87654)
        self.process.output_pipeline_fps = SimpleNamespace(value=24.76543)
        self.process.average_model_interval = SimpleNamespace(value=0.013579)
        self.process.cache_hit_ratio = SimpleNamespace(value=0.4321)
        self.process.gpu_cache_hit_ratio = SimpleNamespace(value=0.6789)
        rng = np.random.default_rng(20260901)
        self.frames = rng.integers(
            0,
            256,
            size=(2, 128, 160, 4),
            dtype=np.uint8,
        )
        self.position = np.array([0.025, -0.01, 0.02, 1.0], dtype=np.float32)

    def tearDown(self):
        for name, value in self.original_args.items():
            setattr(args, name, value)

    def _configure(self, mode, alpha_split, transformed):
        args.output_debug = mode == 'debug'
        args.output_virtual_cam = mode == 'virtual'
        args.output_spout2 = mode == 'spout'
        args.alpha_split = alpha_split
        args.extend_movement = transformed
        args.bongo = transformed
        args.max_ram_cache_len = 2.0
        args.max_gpu_cache_len = 2.0

    def _legacy_reference(self):
        transform = build_output_transform(
            self.position,
            self.frames[0].shape,
            args.extend_movement,
            args.bongo,
        )
        results = []
        for output_image in self.frames:
            bgra_image = apply_output_transform(
                output_image,
                transform,
                copy_identity=args.output_debug,
            )
            if args.output_debug:
                self.process._draw_debug_overlay(bgra_image)

            if args.alpha_split:
                rgba_image = cv2.cvtColor(bgra_image, cv2.COLOR_BGRA2RGBA)
                alpha_channel = rgba_image[:, :, 3]
                rgb_channels = rgba_image[:, :, :3]
                alpha_image = cv2.cvtColor(alpha_channel, cv2.COLOR_GRAY2RGB)
                rgb_channels = cv2.hconcat([rgb_channels, alpha_image])

            if args.output_debug:
                if args.alpha_split:
                    result = cv2.cvtColor(rgb_channels, cv2.COLOR_RGB2BGR)
                else:
                    result = cv2.cvtColor(bgra_image, cv2.COLOR_BGRA2BGR)
            elif args.output_virtual_cam:
                if not args.alpha_split:
                    rgb_channels = cv2.cvtColor(bgra_image, cv2.COLOR_BGRA2RGB)
                result = rgb_channels
            else:
                result = cv2.cvtColor(bgra_image, cv2.COLOR_BGRA2RGBA)
            results.append(result)
        return results

    def test_all_output_modes_match_legacy_bytes_with_reused_buffers(self):
        scenarios = (
            ('debug', False, False),
            ('debug', False, True),
            ('debug', True, True),
            ('virtual', False, False),
            ('virtual', True, True),
            ('spout', False, True),
        )
        for mode, alpha_split, transformed in scenarios:
            with self.subTest(
                mode=mode,
                alpha_split=alpha_split,
                transformed=transformed,
            ):
                self._configure(mode, alpha_split, transformed)
                source_before = self.frames.copy()
                expected = self._legacy_reference()
                destinations = [np.empty_like(frame) for frame in expected]
                guards = [_Guard() for _ in expected]
                bgra_work = np.empty_like(self.frames[0])
                rgba_work = np.empty_like(self.frames[0]) if alpha_split else None

                self.process.post_process_into(
                    self.position,
                    self.frames,
                    destinations,
                    guards,
                    bgra_work,
                    rgba_work,
                )

                for actual, reference in zip(destinations, expected):
                    np.testing.assert_array_equal(actual, reference)
                np.testing.assert_array_equal(self.frames, source_before)
                self.assertTrue(all(guard.entries == 1 for guard in guards))

    def test_compatibility_wrapper_matches_live_destination_path(self):
        self._configure('debug', False, True)
        expected = self._legacy_reference()

        actual = self.process.post_process_ret(self.position, self.frames)

        for frame, reference in zip(actual, expected):
            np.testing.assert_array_equal(frame, reference)


if __name__ == '__main__':
    unittest.main()
