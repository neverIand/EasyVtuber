import os
from pathlib import Path
import subprocess
import sys
import unittest


class LauncherUiTests(unittest.TestCase):
    def test_tabs_presets_and_conditional_controls_layout(self):
        project_root = Path(__file__).resolve().parents[1]
        body = r'''
import wx
import launcher2

app = wx.App(False)
frame = launcher2.MainFrame(None)
panel = frame.panel

assert [
    panel.notebook.GetPageText(index)
    for index in range(panel.notebook.GetPageCount())
] == ['基本设置', '性能与安全', '高级设置']
assert panel.optionSections['frame_rate_limit'] == 'performance'
assert panel.optionDict['safety_preset'].control.GetItems() == [
    '保守（低温优先）',
    '平衡（推荐）',
    '性能（高吞吐）',
    '自定义（调整下方两项）',
]

model_before = panel.optionDict['model_select'].GetValue()
panel.optionDict['safety_preset'].SetValue('Conservative')
panel.OnSafetyPresetChoice()
assert panel.optionDict['frame_rate_limit'].GetValue() == '24'
assert panel.optionDict['gpu_duty_limit'].GetValue() == '70'
assert panel.optionDict['model_select'].GetValue() == model_before

panel.ApplyDefaultSettings()
assert panel.optionDict['safety_preset'].GetValue() == 'Balanced'
assert panel.optionDict['frame_rate_limit'].GetValue() == '30'
assert panel.optionDict['gpu_duty_limit'].GetValue() == '80'
assert panel.optionDict['model_select'].GetValue() == 'seperable_half'

panel.optionDict['frame_rate_limit'].SetValue('60')
panel.OnPacingSettingChoice()
assert panel.optionDict['safety_preset'].GetValue() == 'Custom'
assert '直接调整下面' in panel.performanceNotice.GetLabel()

panel.optionDict['use_tensorrt'].SetValue(False)
panel.OnBackendChoice()
assert panel.optionDict['dml_device'].IsShown()
assert not panel.optionDict['vram_cache_size'].IsShown()

panel.optionDict['ram_cache_size'].SetValue('0b')
panel.OnCacheSettingsChanged()
assert not panel.optionDict['ram_cache_mode'].IsShown()

# Exercise the supported narrow window and every tab at the process's real
# Windows DPI scale. Hidden controls must not prevent any page from laying out.
frame.SetClientSize(frame.FromDIP(wx.Size(680, 520)))
for index in range(panel.notebook.GetPageCount()):
    panel.notebook.SetSelection(index)
    frame.Layout()
    panel._layout_options()
    page = panel.notebook.GetPage(index)
    assert page.GetVirtualSize().width >= 0
    assert page.GetVirtualSize().height >= 0

frame.Destroy()
app.Destroy()
'''
        result = subprocess.run(
            [sys.executable, '-c', body],
            cwd=project_root,
            env={
                **os.environ,
                'PATH': os.pathsep.join((
                    str(
                        project_root
                        / 'envs'
                        / 'TensorRT-RTX-1.3.0.35_cu129'
                        / 'bin'
                    ),
                    str(Path(sys.executable).parent),
                    str(Path(sys.executable).parent / 'Scripts'),
                    str(Path(sys.executable).parent / 'Library' / 'bin'),
                    os.environ.get('PATH', ''),
                )),
            },
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg='stdout:\n{}\nstderr:\n{}'.format(
                result.stdout,
                result.stderr,
            ),
        )


if __name__ == '__main__':
    unittest.main()
