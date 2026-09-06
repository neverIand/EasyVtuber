"""Bounded, single-engine runtime-cache probe; never builds an engine.

The parent monitors NVML and deadlines outside the native worker. Each case
gets a new directory; existing engines and seed caches are read-only inputs.
"""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def worker(args):
    case = Path(args.case_dir)
    durations = {}

    def stage(name, function):
        write_json(case / "stage.json", {"name": name, "started": time.monotonic()})
        print(f"BEGIN {name}", flush=True)
        started = time.perf_counter()
        value = function()
        durations[name] = (time.perf_counter() - started) * 1000
        print(f"END {name} {durations[name]:.3f} ms", flush=True)
        return value

    if args.self_test:
        # PyDLL keeps the worker GIL held. The parent must still time it out.
        stage("native_gil_block", lambda: ctypes.PyDLL("kernel32.dll").Sleep(60000))
        return

    import numpy as np
    import pycuda.driver as cuda
    import tensorrt_rtx as trt

    cuda.init()
    cudart = ctypes.CDLL("cudart64_12.dll")
    if cudart.cudaSetDevice(args.device) != 0:
        raise RuntimeError("cudaSetDevice failed")
    device = cuda.Device(args.device)

    def create_cuda_context():
        if args.context == "application":
            os.environ["EZVTB_DEVICE_ID"] = str(args.device)
            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ezvtuber-rt"))
            import ezvtb_rt
            getattr(ezvtb_rt, "CoreTRT")  # Exercise the real lazy backend bootstrap.
            return None  # The package's atexit handler owns this stack entry.
        if args.context == "primary":
            context = device.retain_primary_context()
            context.push()
            return context
        return device.make_context()

    cuda_context = stage("cuda_context", create_cuda_context)
    if args.pipeline:
        from probe_runtime_pipeline import run_pipeline
        run_pipeline(args, case, stage, durations)
        write_json(case / "stage.json", {"name": "shutdown", "started": time.monotonic()})
        import gc
        gc.collect()
        if cuda_context is not None:
            cuda_context.pop()
            cuda_context.detach()
        print("COMPLETE", flush=True)
        return
    logger = trt.Logger(trt.Logger.INFO)
    runtime = trt.Runtime(logger)
    engine_bytes = Path(args.engine).read_bytes()
    engine_sha256 = hashlib.sha256(engine_bytes).hexdigest()
    if args.implementation == "application":
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ezvtuber-rt"))
        from ezvtb_rt.trt_utils import _deserialize_engine
        from ezvtb_rt.trt_cache import load_runtime_cache
        # Match application ownership: no caller-held Runtime or input blob.
        del engine_bytes
        runtime = None
        engine = stage("deserialize_engine", lambda: _deserialize_engine(Path(args.engine)))
    else:
        engine = stage("deserialize_engine", lambda: runtime.deserialize_cuda_engine(engine_bytes))
    if engine is None:
        raise RuntimeError("Engine deserialization failed; rebuilding is prohibited")
    config = engine.create_runtime_config()
    config.cuda_graph_strategy = (trt.CudaGraphStrategy.WHOLE_GRAPH_CAPTURE
                                  if args.graph == "on" else trt.CudaGraphStrategy.DISABLED)
    cache = None
    loaded = False
    if args.runtime_cache != "off":
        if args.implementation == "application":
            cache, loaded = stage("load_runtime_cache", lambda: load_runtime_cache(config, case / "seed.runtime.cache"))
        else:
            cache = stage("create_runtime_cache", config.create_runtime_cache)
            if args.runtime_cache == "load":
                cache_bytes = (case / "seed.runtime.cache").read_bytes()
                loaded = stage("deserialize_runtime_cache", lambda: cache.deserialize(cache_bytes))
            if config.set_runtime_cache(cache) is False:
                raise RuntimeError("Runtime cache attachment failed")
        if args.runtime_cache == "load" and not loaded:
            raise RuntimeError("Runtime cache was rejected")

    context = stage("create_execution_context", lambda: engine.create_execution_context(config))
    if context is None:
        raise RuntimeError("Execution context creation failed")
    # Startup calls cannot be subdivided. Apply 80% duty cooldown after each.
    time.sleep(sum(durations.values()) / 1000 * 0.25)
    stream = cuda.Stream()
    buffers = []
    input_digests, output_digests = {}, {}
    rng = np.random.default_rng(20260905)
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = tuple(context.get_tensor_shape(name))
        if any(d <= 0 for d in shape):
            raise RuntimeError(f"Probe expects static shapes: {name} {shape}")
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        host = cuda.pagelocked_empty(shape, dtype)
        allocation = cuda.mem_alloc(host.nbytes)
        is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        if is_input:
            if np.issubdtype(dtype, np.floating):
                host[:] = rng.uniform(-0.25, 0.25, shape).astype(dtype)
            else:
                host[:] = rng.integers(0, 256, shape, dtype=dtype)
            input_digests[name] = hashlib.sha256(host.tobytes()).hexdigest()
            cuda.memcpy_htod_async(allocation, host, stream)
        if context.set_tensor_address(name, int(allocation)) is False:
            raise RuntimeError(f"Tensor address rejected: {name}")
        buffers.append((name, host, allocation, is_input))

    def infer():
        if context.execute_async_v3(stream.handle) is False:
            raise RuntimeError("Inference enqueue failed")

    stage("enqueue", infer)
    stage("synchronize", stream.synchronize)
    time.sleep((durations["enqueue"] + durations["synchronize"]) / 1000 * 0.25)
    for name, host, allocation, is_input in buffers:
        if not is_input:
            cuda.memcpy_dtoh_async(host, allocation, stream)
    stage("download", stream.synchronize)
    for name, host, _, is_input in buffers:
        if not is_input:
            if np.issubdtype(host.dtype, np.floating) and not np.isfinite(host).all():
                raise RuntimeError(f"Non-finite output: {name}")
            output_digests[name] = hashlib.sha256(host.tobytes()).hexdigest()

    cache_size = 0
    if cache is not None:
        serialized = stage("serialize_runtime_cache", cache.serialize)
        if serialized is None:
            raise RuntimeError("Runtime cache serialization returned None")
        payload = stage("copy_serialized_bytes", lambda: bytes(serialized))
        cache_size = len(payload)
        (case / "output.runtime.cache").write_bytes(payload)
        del serialized, payload

    result = {"status": "ok", "trt_version": trt.__version__, "gpu": device.name(),
              "context": args.context, "graph": args.graph, "runtime_cache": args.runtime_cache,
              "cache_loaded": bool(loaded), "cache_bytes": cache_size, "durations_ms": durations,
              "input_sha256": input_digests, "output_sha256": output_digests,
              "implementation": args.implementation, "engine_sha256": engine_sha256}
    write_json(case / "worker.json", result)
    # Keep all factories alive until their dependents have been released.
    stage("final_synchronize", stream.synchronize)
    write_json(case / "stage.json", {"name": "shutdown", "started": time.monotonic()})
    for _, _, allocation, _ in buffers:
        allocation.free()
    del buffers, allocation, host, context, config, cache, engine, runtime, stream
    import gc
    gc.collect()
    if cuda_context is not None:
        cuda_context.pop()
        cuda_context.detach()
    print("COMPLETE", flush=True)


def supervise(args):
    from performance_compare import NvmlSampler

    case = Path(args.case_dir).resolve()
    case.mkdir(parents=True, exist_ok=False)
    if not args.self_test and not args.pipeline:
        if not Path(args.engine).is_file() or Path(args.engine).suffix.lower() != ".trt":
            raise ValueError("An existing .trt engine is required; building is prohibited")
        if args.runtime_cache == "load":
            if args.seed_cache is None:
                raise ValueError("--seed-cache is required for load")
            shutil.copyfile(args.seed_cache, case / "seed.runtime.cache")
    if args.pipeline and args.runtime_cache == "load":
        if args.seed_cache is None or not Path(args.seed_cache).is_dir():
            raise ValueError("Pipeline load requires --seed-cache DIRECTORY")
        target = case / "runtime-cache"
        target.mkdir()
        for name in ("decomposer", "combiner", "morpher", "rotator", "editor"):
            shutil.copyfile(Path(args.seed_cache) / f"{name}.runtime.cache", target / f"{name}.runtime.cache")

    sampler = None
    if not args.self_test:
        sampler = NvmlSampler(device_index=args.device, temperature_limit=args.max_temperature)
        sampler.start()
        time.sleep(0.1)
        if sampler.error or not sampler.samples:
            sampler.stop()
            raise RuntimeError("NVML monitoring is unavailable")
        if sampler.samples[-1]["temperature_c"] >= 65:
            sampler.stop()
            raise RuntimeError("GPU must cool below 65 C before a probe")

    env = os.environ.copy()
    python_root = Path(sys.executable).parent
    trt_bin = Path(args.trt_bin).resolve() if args.trt_bin else next(python_root.parent.glob("TensorRT-RTX*/bin"))
    env["PATH"] = os.pathsep.join([str(python_root), str(python_root / "Library/bin"), str(trt_bin), env.get("PATH", "")])
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", "--case-dir", str(case),
               "--engine", str(Path(args.engine).resolve()) if args.engine else "unused",
               "--context", args.context, "--runtime-cache", args.runtime_cache, "--graph", args.graph,
               "--device", str(args.device), "--implementation", args.implementation]
    if args.self_test:
        command.append("--self-test")
    if args.pipeline:
        command.append("--pipeline")
    process = None
    started = time.monotonic()
    reason, last_stage = None, None
    stage = {"name": "worker_startup", "started": started}
    try:
        with (case / "worker.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
            while process.poll() is None:
                try:
                    stage = json.loads((case / "stage.json").read_text())
                except (FileNotFoundError, ValueError, PermissionError):
                    # An atomic replacement can be briefly unreadable on
                    # Windows. Keep the last observed deadline in that case.
                    pass
                if stage["name"] != last_stage:
                    last_stage = stage["name"]
                    print(f"PROBE {last_stage}", flush=True)
                if sampler and (sampler.error or sampler.stop_reason):
                    reason = sampler.error or sampler.stop_reason
                elif time.monotonic() - stage["started"] > args.stage_timeout:
                    reason = f"stage_timeout: {stage['name']}"
                elif time.monotonic() - started > args.total_timeout:
                    reason = "total_timeout"
                if reason:
                    process.kill()
                    break
                time.sleep(0.05)
            returncode = process.wait(timeout=5)
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        if sampler:
            sampler.stop()
    result = {"status": "ok" if returncode == 0 and reason is None else "failed",
              "reason": reason, "returncode": returncode, "worker_pid": process.pid,
              "worker_exited": process.poll() is not None, "last_stage": last_stage,
              "elapsed_seconds": time.monotonic() - started,
              "gpu": sampler.summary() if sampler else None}
    if (case / "worker.json").is_file():
        result["worker"] = json.loads((case / "worker.json").read_text())
    if sampler:
        write_json(case / "telemetry.json", sampler.samples)
    write_json(case / "result.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result["status"] == "ok" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", required=True)
    parser.add_argument("--engine")
    parser.add_argument("--context", choices=("primary", "user", "application"), default="primary")
    parser.add_argument("--implementation", choices=("native", "application"), default="native")
    parser.add_argument("--runtime-cache", choices=("off", "fresh", "load"), default="off")
    parser.add_argument("--seed-cache")
    parser.add_argument("--graph", choices=("on", "off"), default="on")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--stage-timeout", type=float, default=20)
    parser.add_argument("--total-timeout", type=float, default=60)
    parser.add_argument("--max-temperature", type=int, default=75)
    parser.add_argument("--trt-bin")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--pipeline", action="store_true", help="Probe all five THA3 engines with the project wrapper")
    parser.add_argument("--self-test", action="store_true", help="Exercise external timeout without GPU")
    args = parser.parse_args()
    if not 0 < args.max_temperature <= 80 or not 0 < args.stage_timeout <= 30 or not 0 < args.total_timeout <= 90:
        parser.error("Safety limits exceeded")
    if args.worker:
        try:
            worker(args)
        except Exception:
            traceback.print_exc()
            return 1
        return 0
    return supervise(args)


if __name__ == "__main__":
    raise SystemExit(main())
