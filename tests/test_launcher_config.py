import tempfile
from pathlib import Path
import unittest

from src.utils.launcher_config import (
    DEFAULT_LAUNCHER_CONFIG,
    LAUNCHER_CONFIG_VERSION,
    apply_safety_preset,
    build_launch_command,
    describe_ram_cache,
    load_launcher_config,
    normalize_launcher_config,
    save_launcher_config,
)


class LauncherConfigTests(unittest.TestCase):
    def test_legacy_preset_migrates_without_changing_quality_controls(self):
        legacy = {
            'preset': 'High',
            'frame_rate_limit': '30',
            'gpu_duty_limit': '90',
            'model_select': 'standard_full',
            'ram_cache_size': '4gb',
            'vram_cache_size': '1gb',
            'cache_simplify': 'Medium',
        }

        migrated = normalize_launcher_config(legacy)

        self.assertEqual(migrated['config_version'], LAUNCHER_CONFIG_VERSION)
        self.assertEqual(migrated['safety_preset'], 'Performance')
        self.assertEqual(migrated['ram_cache_mode'], 'raw')
        self.assertEqual(migrated['model_select'], 'standard_full')
        self.assertEqual(migrated['ram_cache_size'], '4gb')
        self.assertEqual(migrated['vram_cache_size'], '1gb')
        self.assertEqual(migrated['cache_simplify'], 'Medium')
        self.assertNotIn('preset', migrated)

    def test_actual_pacing_repairs_a_conflicting_saved_preset_label(self):
        migrated = normalize_launcher_config({
            'safety_preset': 'Performance',
            'frame_rate_limit': 24,
            'gpu_duty_limit': 70,
        })

        self.assertEqual(migrated['safety_preset'], 'Conservative')
        self.assertEqual(migrated['frame_rate_limit'], '24')
        self.assertEqual(migrated['gpu_duty_limit'], '70')

    def test_safety_preset_only_changes_pacing(self):
        original = {
            **DEFAULT_LAUNCHER_CONFIG,
            'model_select': 'standard_full',
            'cache_simplify': 'Low',
            'ram_cache_size': '8gb',
            'sr': 'waifu2x_x2_full',
        }

        updated = apply_safety_preset(original, 'Conservative')

        self.assertEqual(updated['frame_rate_limit'], '24')
        self.assertEqual(updated['gpu_duty_limit'], '70')
        for key in ('model_select', 'cache_simplify', 'ram_cache_size', 'sr'):
            self.assertEqual(updated[key], original[key])

    def test_raw_cache_estimates_make_resolution_tradeoff_explicit(self):
        base_only = describe_ram_cache('2gb', 'raw')
        with_sr = describe_ram_cache('2gb', 'raw', super_resolution=True)

        self.assertIn('2048', base_only)
        self.assertIn('512²', base_only)
        self.assertIn('409', with_sr)
        self.assertIn('1024²', with_sr)
        self.assertIn('1:4', with_sr)

    def test_brotli_and_disabled_cache_descriptions_do_not_overpromise(self):
        brotli = describe_ram_cache('2gb', 'brotli')
        disabled = describe_ram_cache('2gb', 'raw', simplify_enabled=False)

        self.assertIn('实际容量取决于画面内容', brotli)
        self.assertIn('不会启用', disabled)

    def test_command_snapshot_includes_ram_storage_mode(self):
        config = {
            **DEFAULT_LAUNCHER_CONFIG,
            'character': 'sanae_char',
            'input': 0,
            'ifm': '192.0.2.5',
            'output': 1,
            'use_tensorrt': False,
            'dml_device': '1',
            'frame_rate_limit': '24',
            'gpu_duty_limit': '70',
            'cache_simplify': 'Medium',
            'ram_cache_size': '2gb',
            'ram_cache_mode': 'raw',
            'vram_cache_size': '1gb',
            'model_select': 'seperable_full',
            'interpolation': 'x3_half',
            'sr': 'waifu2x_x2_half',
            'is_extend_movement': True,
            'is_eyebrow': True,
            'min_cutoff': 50,
            'beta': 50,
        }

        command = build_launch_command(
            config,
            r'C:\Python\python.exe',
            (2560, 1440),
        )

        self.assertEqual(command, [
            r'C:\Python\python.exe', '-m', 'src.main',
            '--character', 'sanae_char',
            '--ifm_input', '192.0.2.5:49983',
            '--breath_cycle', 'inf',
            '--output_virtual_cam',
            '--extend_movement',
            '--eyebrow',
            '--simplify', '2',
            '--cache', '2gb',
            '--ram_cache_mode', 'raw',
            '--gpu_cache', '1gb',
            '--use_interpolation',
            '--interpolation_half',
            '--interpolation_scale', '3',
            '--model_version', 'v3',
            '--model_seperable',
            '--frame_rate_limit', '24',
            '--gpu_duty_limit', '70',
            '--dml_device_id', '1',
            '--use_sr',
            '--sr_half',
            '--filter_min_cutoff', '25.0',
            '--filter_beta', '0.25',
        ])

    def test_config_save_is_normalized_atomic_and_round_trips(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'launcher.json'
            saved = save_launcher_config({
                **DEFAULT_LAUNCHER_CONFIG,
                'frame_rate_limit': '24',
                'gpu_duty_limit': '70',
                'ram_cache_mode': 'brotli',
            }, path)
            loaded = load_launcher_config(path)

            self.assertEqual(saved, loaded)
            self.assertEqual(loaded['safety_preset'], 'Conservative')
            self.assertEqual(loaded['ram_cache_mode'], 'brotli')
            self.assertFalse(Path(str(path) + '.tmp').exists())


if __name__ == '__main__':
    unittest.main()
