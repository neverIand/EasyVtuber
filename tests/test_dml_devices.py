from types import SimpleNamespace
import unittest

from src.utils.dml_devices import (
    DirectMLAdapter,
    discover_directml_adapters,
    launcher_directml_choices,
    preferred_directml_adapter,
    select_directml_adapter,
)


def _ep_device(name, options, description, discrete, high_performance, vendor):
    return SimpleNamespace(
        ep_name=name,
        ep_options=options,
        device=SimpleNamespace(
            metadata={
                'Description': description,
                'Discrete': discrete,
                'DxgiHighPerformanceIndex': high_performance,
            },
            vendor=vendor,
        ),
    )


class DirectMLDeviceTests(unittest.TestCase):
    def setUp(self):
        self.fake_ort = SimpleNamespace(
            get_ep_devices=lambda: [
                _ep_device(
                    'CPUExecutionProvider', {}, 'CPU', None, None, 'Intel'),
                _ep_device(
                    'DmlExecutionProvider', {'device_id': '1'},
                    'NVIDIA RTX', '1', '0', 'NVIDIA'),
                _ep_device(
                    'DmlExecutionProvider', {'device_id': '0'},
                    'Intel UHD', '0', '1', 'Intel'),
            ]
        )

    def test_discovers_actual_provider_device_ids(self):
        adapters = discover_directml_adapters(self.fake_ort)

        self.assertEqual([adapter.device_id for adapter in adapters], [0, 1])
        self.assertEqual(adapters[0].description, 'Intel UHD')
        self.assertFalse(adapters[0].discrete)
        self.assertTrue(adapters[1].discrete)

    def test_auto_prefers_discrete_adapter(self):
        adapters = discover_directml_adapters(self.fake_ort)

        self.assertEqual(preferred_directml_adapter(adapters).device_id, 1)
        self.assertEqual(
            select_directml_adapter(adapters=adapters).device_id,
            1,
        )

    def test_explicit_adapter_is_preserved(self):
        adapters = discover_directml_adapters(self.fake_ort)

        selected = select_directml_adapter(0, adapters=adapters)

        self.assertEqual(selected.device_id, 0)
        self.assertIn('Intel UHD', selected.display_label)

    def test_invalid_explicit_adapter_is_rejected(self):
        adapters = discover_directml_adapters(self.fake_ort)

        with self.assertRaisesRegex(ValueError, 'available device IDs: 0, 1'):
            select_directml_adapter(4, adapters=adapters)

    def test_launcher_choices_do_not_need_onnxruntime(self):
        labels, mappings = launcher_directml_choices(
            ['Intel UHD', 'NVIDIA RTX'])

        self.assertEqual(mappings, ['auto', '0', '1'])
        self.assertIn('推荐', labels[0])
        self.assertEqual(labels[2], 'GPU 1: NVIDIA RTX')

    def test_high_performance_index_breaks_unknown_device_ties(self):
        adapters = [
            DirectMLAdapter(0, 'GPU A', None, 1),
            DirectMLAdapter(1, 'GPU B', None, 0),
        ]

        self.assertEqual(preferred_directml_adapter(adapters).device_id, 1)


if __name__ == '__main__':
    unittest.main()
