"""Short real ModelClientProcess startup/IPC check with external NVML cutoff.

Uses an isolated cache directory containing copies of existing engines. Does
not start a launcher, camera, network input, Spout sender, or new engine build.
"""

import argparse
import hashlib
import json
import multiprocessing
from multiprocessing import shared_memory
import os
from pathlib import Path
import shutil
import sys
import threading
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-dir', required=True, type=Path)
    parser.add_argument('--seed-dir', required=True, type=Path)
    parser.add_argument('--cache-mode', choices=('off', 'load'), default='load')
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    case = args.case_dir.resolve()
    case.mkdir(parents=True, exist_ok=False)
    cache_dir = case / 'trt-cache'
    cache_dir.mkdir()
    audit = json.loads((project / 'benchmark_results/2026-09-05/runtime-cache-audit.json').read_text())
    for model in audit['models']:
        source = Path(model['engine_path'])
        if not source.is_file():
            raise RuntimeError('Existing engine missing; engine builds are prohibited')
        shutil.copyfile(source, cache_dir / source.name)
        if args.cache_mode == 'load':
            shutil.copyfile(args.seed_dir / (model['name'] + '.runtime.cache'),
                            cache_dir / Path(model['runtime']['2']['path']).name)

    python_root = Path(sys.executable).parent
    trt_bin = next(python_root.parent.glob('TensorRT-RTX*/bin'))
    os.environ['PATH'] = os.pathsep.join([str(python_root), str(python_root / 'Library/bin'),
                                        str(trt_bin), os.environ.get('PATH', '')])
    os.environ['EZVTB_TRT_CACHE_DIR'] = str(cache_dir)
    os.environ['EZVTB_TRT_RUNTIME_CACHE'] = '1' if args.cache_mode == 'load' else ''
    os.environ['EZVTB_TRT_REQUIRE_ENGINE_CACHE'] = '1'
    os.environ['EZVTB_DEVICE_ID'] = '0'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['PYTHONUNBUFFERED'] = '1'
    sys.path.insert(0, str(project))
    # Spawn copies sys.argv; children need application flags, not probe flags.
    sys.argv = [sys.argv[0], '--use_tensorrt', '--model_seperable', '--output_spout2',
                '--cache', '0mb', '--gpu_cache', '0mb', '--simplify', '0',
                '--gpu_duty_limit', '80', '--frame_rate_limit', '10']

    import cv2
    import numpy as np
    from PIL import Image
    from performance_compare import NvmlSampler
    from src.model_infer_client import ModelClientProcess
    from src.utils.model_startup import start_model_process, stop_failed_startup
    from src.utils.preprocess import clear_transparent_rgb
    from src.utils.shared_mem_guard import SharedMemoryGuard

    sampler = NvmlSampler(temperature_limit=75)
    sampler.start()
    time.sleep(0.1)
    if sampler.error or not sampler.samples or sampler.samples[-1]['temperature_c'] >= 65:
        sampler.stop()
        raise RuntimeError('Need available NVML and a GPU below 65 C')
    image = clear_transparent_rgb(Image.open(project / 'data/images/lambda_00.png').convert('RGBA'))
    input_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGRA)
    pose_memory = shared_memory.SharedMemory(create=True, size=49 * 4)
    pose_memory.buf[:] = bytes(49 * 4)
    input_fps = multiprocessing.Value('f', 10)
    workers = []
    finished, safety_stop = threading.Event(), threading.Event()
    safety_reason = []
    overall_started = time.monotonic()

    def factory(**options):
        if safety_stop.is_set():
            raise RuntimeError(safety_reason[0])
        process = ModelClientProcess(input_image, pose_memory, input_fps, **options)
        workers.append(process)
        return process

    def watch():
        while not finished.wait(0.05):
            reason = sampler.error or sampler.stop_reason
            if time.monotonic() - overall_started > 75:
                reason = 'External 75s total deadline'
            if reason:
                safety_reason.append(reason)
                safety_stop.set()
                for process in workers:
                    if process.pid and process.is_alive():
                        process.terminate()
                return

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    process = None
    result = {}
    try:
        started = time.perf_counter()
        # Guard the off baseline too: this check always prohibits builds and
        # needs readiness before inspecting IPC. Its environment keeps cache off.
        process = start_model_process(factory, runtime_cache_trial=True, timeout_seconds=30)
        result['startup_seconds'] = time.perf_counter() - started
        result['fallback_used'] = process.disable_runtime_cache
        # require_engine_cache also forbids get_core's usual ORT fallback.
        result['backend'] = 'TensorRT'
        if not process.finish_event.wait(timeout=5):
            raise RuntimeError('No output frame after readiness')
        guard = SharedMemoryGuard(process.ret_shared_mem, 'ret_shm_ctrl_batch_0')
        if not guard.acquire(timeout_ms=1000):
            raise RuntimeError('Output mutex acquisition timed out')
        try:
            frame = np.ndarray((512, 512, 4), dtype=np.uint8, buffer=process.ret_shared_mem.buf).copy()
        finally:
            guard.release()
        result['frame_rgba_sha256'] = hashlib.sha256(frame.tobytes()).hexdigest()
        result['status'] = 'ok'
    except Exception as error:
        result.update(status='failed', error=str(error))
    finally:
        finished.set()
        watcher.join(timeout=2)
        for worker in workers:
            if worker.is_alive():
                stop_failed_startup(worker)
        # A successful process could exit between the try body and cleanup.
        if process is not None and not process.ret_shared_mem._buf is None:
            process.ret_shared_mem.close()
            process.ret_shared_mem.unlink()
        pose_memory.close()
        pose_memory.unlink()
        sampler.stop()
    result.update(cache_mode=args.cache_mode, workers=[{'pid': w.pid, 'exited': not w.is_alive()} for w in workers],
                  gpu=sampler.summary(), safety_reason=safety_reason)
    (case / 'telemetry.json').write_text(json.dumps(sampler.samples, indent=2) + '\n', encoding='utf-8')
    (case / 'result.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result['status'] == 'ok' and not safety_reason else 1


if __name__ == '__main__':
    raise SystemExit(main())
