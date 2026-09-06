"""Supervise runtime-cache startup outside the inference process."""

import time


def stop_failed_startup(process):
    """Finish the old worker before releasing its output or starting a retry."""
    if process.is_alive():
        process.terminate()
    process.join(timeout=5)
    if process.is_alive():
        process.kill()
        process.join(timeout=5)
    if process.is_alive():
        raise RuntimeError('Cannot stop the failed model worker; refusing overlapping GPU workers')
    process.ret_shared_mem.close()
    process.ret_shared_mem.unlink()


def _spawn_model_process(factory, *, disable_runtime_cache, require_engine_cache):
    process = factory(disable_runtime_cache=disable_runtime_cache,
                      require_engine_cache=require_engine_cache)
    process.daemon = True
    try:
        process.start()
    except BaseException:
        process.ret_shared_mem.close()
        process.ret_shared_mem.unlink()
        raise
    return process


def start_model_process(factory, *, runtime_cache_trial=False, timeout_seconds=60.0,
                        allow_engine_build=False):
    """Retry one failed cache-enabled startup with caching disabled.

    The factory accepts disable_runtime_cache and require_engine_cache keyword
    arguments. Both guarded attempts use existing engines only. If the first
    worker explicitly reports a required engine build and allow_engine_build
    is set, start the normal asynchronous build path with runtime cache off.
    Timeouts and other failures can never enable engine building.
    """
    if not runtime_cache_trial:
        return _spawn_model_process(factory, disable_runtime_cache=False,
                                    require_engine_cache=False)
    if timeout_seconds <= 0:
        raise ValueError('Startup timeout must be positive')

    for disable_cache in (False, True):
        process = _spawn_model_process(factory, disable_runtime_cache=disable_cache,
                                       require_engine_cache=True)
        started = time.monotonic()
        worker_exited = False
        try:
            while True:
                remaining = timeout_seconds - (time.monotonic() - started)
                if remaining <= 0:
                    reason = f'initialization exceeded {timeout_seconds:g}s'
                    break
                if process.ready_event.wait(timeout=min(0.1, remaining)) and process.is_alive():
                    return process
                if not process.is_alive():
                    worker_exited = True
                    reason = f'worker exited with code {process.exitcode}'
                    break
        except BaseException:
            stop_failed_startup(process)
            raise
        needs_engine_build = (allow_engine_build and not disable_cache
                              and worker_exited
                              and process.engine_cache_required_event.is_set())
        stop_failed_startup(process)
        if needs_engine_build:
            print(
                'TensorRT requires an engine build. Starting the normal engine-build '
                'path with runtime cache disabled for this run.',
                flush=True,
            )
            return _spawn_model_process(factory, disable_runtime_cache=True,
                                        require_engine_cache=False)
        if disable_cache:
            raise RuntimeError(f'TensorRT startup failed with runtime cache disabled: {reason}')
        print(
            f'TensorRT runtime-cache startup failed ({reason}). '
            'Retrying once with runtime cache disabled; existing engines are preserved.',
            flush=True,
        )
