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

    def test_runtime_cache_trial_cannot_silently_switch_backend(self):
        self.run_without_torch(r'''
from unittest import mock
from src import ezvtb_rt_interface as interface

class UnavailableTRT:
    CoreORT = mock.Mock()

    @property
    def CoreTRT(self):
        raise ImportError('missing TensorRT DLL')

runtime = UnavailableTRT()
with mock.patch.object(interface, 'ezvtb_rt', runtime), mock.patch.object(interface.args, 'use_tensorrt', True):
    try:
        interface.get_core(use_tensorrt=True, allow_backend_fallback=False)
    except RuntimeError as error:
        assert isinstance(error.__cause__, ImportError)
    else:
        raise AssertionError('Guarded trial silently switched backend')
    runtime.CoreORT.assert_not_called()
    assert interface.args.use_tensorrt

    # Preserve the existing fallback policy outside guarded trials.
    interface.get_core(use_tensorrt=True)
    runtime.CoreORT.assert_called_once()
    assert not interface.args.use_tensorrt
''')

    def test_model_worker_reports_only_typed_engine_cache_failures(self):
        self.run_without_torch(r'''
from unittest import mock
from src.model_infer_client import ModelClientProcess
from ezvtb_rt.trt_cache import EngineCacheRequiredError

for error, expected_signal in [(EngineCacheRequiredError('missing engine'), True),
                               (RuntimeError('runtime cache failed'), False)]:
    worker = object.__new__(ModelClientProcess)
    worker.engine_cache_required_event = mock.Mock()
    worker._run_inference = mock.Mock(side_effect=error)
    try:
        worker.run()
    except type(error):
        pass
    else:
        raise AssertionError('Startup failure was swallowed')
    assert worker.engine_cache_required_event.set.called == expected_signal
''')


if __name__ == '__main__':
    unittest.main()
