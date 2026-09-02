import os
import subprocess
import sys
from pathlib import Path
import unittest


class OptionalDependencyTests(unittest.TestCase):
    def run_without_torch(self, body):
        project_root = Path(__file__).resolve().parents[1]
        blocker = r'''
import importlib.abc
import sys


class TorchBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise ModuleNotFoundError('torch intentionally unavailable')
        return None


sys.meta_path.insert(0, TorchBlocker())
'''
        result = subprocess.run(
            [sys.executable, '-c', blocker + body],
            cwd=project_root,
            env={
                **os.environ,
                'PATH': os.pathsep.join(
                    (
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
                    )
                ),
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

    def test_launcher_imports_without_torch(self):
        self.run_without_torch(r'''
import launcher2

assert isinstance(launcher2.hasTRTSupport, bool)
assert 'torch' not in sys.modules
assert 'onnxruntime' not in sys.modules
assert launcher2.dmlDeviceMapping[0] == 'auto'
''')

    def test_runtime_and_ifacialmocap_import_without_torch(self):
        self.run_without_torch(r'''
import src.main
from tha2.poser.modes.mode_20_wx import IFacialMocapPoseConverter20
from src.ezvtb_rt_interface import ezvtb_rt

assert IFacialMocapPoseConverter20().pose_size == 45
assert ezvtb_rt.CoreORT is not None
assert 'torch' not in sys.modules
''')


if __name__ == '__main__':
    unittest.main()
