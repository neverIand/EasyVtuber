import multiprocessing
import os
import subprocess
import sys
from pathlib import Path


def report():
    print('multiprocessing child:', os.environ.get('EZVTB_PROBE_ENV'), os.environ.get('EZVTB_TRT_RUNTIME_CACHE'), os.environ.get('PATH')[:200], flush=True)


if __name__ == '__main__':
    os.environ['EZVTB_PROBE_ENV'] = 'set-in-parent'
    python_root = Path(sys.executable).parent
    trt_bin = next(python_root.parent.glob('TensorRT-RTX*/bin'))
    os.environ['PATH'] = os.pathsep.join([str(python_root), str(python_root / 'Library/bin'), str(trt_bin), os.environ.get('PATH', '')])
    os.environ['EZVTB_TRT_RUNTIME_CACHE'] = '1'
    os.environ['EZVTB_TRT_REQUIRE_ENGINE_CACHE'] = '1'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['PYTHONUNBUFFERED'] = '1'
    sys.argv = [sys.argv[0], '--use_tensorrt', '--model_seperable', '--output_spout2', '--cache', '0mb', '--gpu_cache', '0mb', '--simplify', '0', '--gpu_duty_limit', '80', '--frame_rate_limit', '10']
    import cv2
    import numpy
    project = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(project))
    sys.path.insert(0, str(project / 'benchmarks'))
    from performance_compare import NvmlSampler
    from src.model_infer_client import ModelClientProcess
    from src.utils.preprocess import clear_transparent_rgb
    from src.utils.shared_mem_guard import SharedMemoryGuard
    sampler = NvmlSampler(temperature_limit=75)
    sampler.start()
    value = multiprocessing.Value('f', 10)
    pose = multiprocessing.shared_memory.SharedMemory(create=True, size=49 * 4)
    print('parent:', os.environ.get('EZVTB_PROBE_ENV'), flush=True)
    subprocess.run([sys.executable, '-c', "import os; print('subprocess child:', os.environ.get('EZVTB_PROBE_ENV'))"])
    worker = multiprocessing.Process(target=report)
    worker.start()
    worker.join(timeout=5)
    if worker.is_alive():
        worker.terminate()
        worker.join(timeout=5)
    pose.close()
    pose.unlink()
    sampler.stop()
    pose = multiprocessing.shared_memory.SharedMemory(create=True, size=49 * 4)
    process = ModelClientProcess(numpy.zeros((512, 512, 4), numpy.uint8), pose, value,
                                 require_engine_cache=True)
    process.run = report
    process.daemon = True
    process.start()
    process.join(timeout=5)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
    process.ret_shared_mem.close()
    process.ret_shared_mem.unlink()
    pose.close()
    pose.unlink()
