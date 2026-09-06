import unittest
from unittest import mock
import ctypes
import multiprocessing
from multiprocessing import shared_memory
import sys
import time

from src.utils.model_startup import start_model_process, stop_failed_startup


def native_startup_worker(ready, blocked):
    if blocked:
        ctypes.PyDLL('kernel32.dll').Sleep(60000)
    else:
        ready.set()
        time.sleep(60)


class NativeWorker(multiprocessing.Process):
    def __init__(self, blocked):
        self.ready_event = multiprocessing.Event()
        self.ret_shared_mem = shared_memory.SharedMemory(create=True, size=1)
        super().__init__(target=native_startup_worker, args=(self.ready_event, blocked))


class FakeWorker:
    def __init__(self, state, events, number):
        self.state, self.events, self.number = state, events, number
        self.alive = False
        self.exitcode = None
        self.ready_event = mock.Mock()
        self.ready_event.wait.side_effect = self.wait
        self.engine_cache_required_event = mock.Mock()
        self.engine_cache_required_event.is_set.return_value = state == 'missing_engine'
        self.ret_shared_mem = mock.Mock()

    def start(self):
        self.events.append(('start', self.number))
        self.alive = True

    def wait(self, timeout):
        if self.state in ('exit', 'missing_engine'):
            self.alive, self.exitcode = False, 7
        return self.state == 'ready'

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.events.append(('terminate', self.number))
        self.alive = False

    def join(self, timeout):
        self.events.append(('join', self.number))

    def kill(self):
        self.terminate()


class ModelStartupTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == 'win32', 'Windows native GIL-block regression')
    def test_real_process_with_blocked_gil_is_reaped_and_retried(self):
        workers = []

        def factory(**options):
            process = NativeWorker(blocked=not options['disable_runtime_cache'])
            workers.append(process)
            return process

        try:
            result = start_model_process(factory, runtime_cache_trial=True, timeout_seconds=2)
            self.assertEqual(len(workers), 2)
            self.assertFalse(workers[0].is_alive())
            self.assertTrue(result.ready_event.is_set())
            stop_failed_startup(result)
            self.assertTrue(all(not process.is_alive() for process in workers))
        finally:
            for process in workers:
                if process.is_alive():
                    stop_failed_startup(process)

    def factory(self, *states):
        events, workers, options = [], [], []

        def create(**kwargs):
            options.append(kwargs)
            worker = FakeWorker(states[len(workers)], events, len(workers))
            workers.append(worker)
            return worker

        return create, events, workers, options

    def test_normal_startup_does_not_wait_or_require_cached_engines(self):
        create, _, workers, options = self.factory('blocked')
        self.assertIs(start_model_process(create), workers[0])
        workers[0].ready_event.wait.assert_not_called()
        self.assertEqual(options, [dict(disable_runtime_cache=False, require_engine_cache=False)])

    def test_failed_worker_is_reaped_before_one_cache_disabled_retry(self):
        create, events, workers, options = self.factory('exit', 'ready')
        result = start_model_process(create, runtime_cache_trial=True)
        self.assertIs(result, workers[1])
        self.assertLess(events.index(('join', 0)), events.index(('start', 1)))
        self.assertEqual(options, [dict(disable_runtime_cache=False, require_engine_cache=True),
                                   dict(disable_runtime_cache=True, require_engine_cache=True)])
        workers[0].ret_shared_mem.close.assert_called_once_with()
        workers[0].ret_shared_mem.unlink.assert_called_once_with()
        workers[1].ret_shared_mem.close.assert_not_called()

    def test_native_block_is_terminated_before_retry(self):
        create, events, workers, _ = self.factory('blocked', 'ready')
        with mock.patch('src.utils.model_startup.time.monotonic', side_effect=[0, 2, 2, 2]):
            result = start_model_process(create, runtime_cache_trial=True, timeout_seconds=1)
        self.assertIs(result, workers[1])
        self.assertLess(events.index(('terminate', 0)), events.index(('start', 1)))

    def test_retry_failure_is_reported_without_a_third_worker(self):
        create, _, workers, _ = self.factory('exit', 'exit')
        with self.assertRaisesRegex(RuntimeError, 'runtime cache disabled'):
            start_model_process(create, runtime_cache_trial=True)
        self.assertEqual(len(workers), 2)
        for worker in workers:
            worker.ret_shared_mem.unlink.assert_called_once_with()

    def test_kill_is_used_when_terminate_does_not_finish(self):
        process = mock.Mock()
        process.is_alive.side_effect = [True, True, False]
        stop_failed_startup(process)
        process.terminate.assert_called_once_with()
        process.kill.assert_called_once_with()
        self.assertEqual(process.join.call_count, 2)
        process.ret_shared_mem.unlink.assert_called_once_with()

    def test_unstoppable_worker_is_not_replaced_or_unmapped(self):
        process = mock.Mock()
        process.ready_event.wait.return_value = False
        process.is_alive.return_value = True
        factory = mock.Mock(return_value=process)
        with mock.patch('src.utils.model_startup.time.monotonic', side_effect=[0, 2]):
            with self.assertRaisesRegex(RuntimeError, 'refusing overlapping'):
                start_model_process(factory, runtime_cache_trial=True, timeout_seconds=1)
        factory.assert_called_once()
        process.ret_shared_mem.close.assert_not_called()

    def test_spawn_failure_releases_output_without_retry(self):
        process = mock.Mock()
        process.start.side_effect = OSError('spawn failed')
        factory = mock.Mock(return_value=process)
        with self.assertRaisesRegex(OSError, 'spawn failed'):
            start_model_process(factory, runtime_cache_trial=True)
        factory.assert_called_once()
        process.ret_shared_mem.close.assert_called_once_with()
        process.ret_shared_mem.unlink.assert_called_once_with()

    def test_explicit_missing_engine_preserves_normal_build_path(self):
        create, events, workers, options = self.factory('missing_engine', 'blocked')
        result = start_model_process(create, runtime_cache_trial=True, allow_engine_build=True)
        self.assertIs(result, workers[1])
        self.assertLess(events.index(('join', 0)), events.index(('start', 1)))
        self.assertEqual(options[1], dict(disable_runtime_cache=True, require_engine_cache=False))
        workers[1].ready_event.wait.assert_not_called()
        workers[0].ret_shared_mem.unlink.assert_called_once_with()

    def test_strict_probe_never_allows_missing_engine_to_build(self):
        create, _, workers, options = self.factory('missing_engine', 'missing_engine')
        with self.assertRaisesRegex(RuntimeError, 'runtime cache disabled'):
            start_model_process(create, runtime_cache_trial=True, allow_engine_build=False)
        self.assertEqual(len(workers), 2)
        self.assertTrue(all(option['require_engine_cache'] for option in options))

    def test_untyped_failure_cannot_request_engine_build(self):
        create, _, _, options = self.factory('exit', 'ready')
        start_model_process(create, runtime_cache_trial=True, allow_engine_build=True)
        self.assertEqual(options[1], dict(disable_runtime_cache=True, require_engine_cache=True))

    def test_timeout_cannot_request_engine_build(self):
        create, _, workers, options = self.factory('blocked', 'ready')
        with mock.patch('src.utils.model_startup.time.monotonic', side_effect=[0, 2, 2, 2]):
            start_model_process(create, runtime_cache_trial=True, allow_engine_build=True, timeout_seconds=1)
        workers[0].engine_cache_required_event.is_set.assert_not_called()
        self.assertEqual(options[1], dict(disable_runtime_cache=True, require_engine_cache=True))

    def test_missing_engine_during_recovery_cannot_start_a_build(self):
        create, _, workers, options = self.factory('exit', 'missing_engine')
        with self.assertRaisesRegex(RuntimeError, 'runtime cache disabled'):
            start_model_process(create, runtime_cache_trial=True, allow_engine_build=True)
        self.assertEqual(len(workers), 2)
        self.assertTrue(all(option['require_engine_cache'] for option in options))

    def test_expired_deadline_cannot_build_from_a_late_missing_signal(self):
        first = mock.Mock()
        first.is_alive.return_value = False
        first.engine_cache_required_event.is_set.return_value = True
        second = FakeWorker('ready', [], 1)
        factory = mock.Mock(side_effect=[first, second])
        with mock.patch('src.utils.model_startup.time.monotonic', side_effect=[0, 2, 2, 2]):
            self.assertIs(start_model_process(factory, runtime_cache_trial=True,
                                             allow_engine_build=True, timeout_seconds=1), second)
        first.engine_cache_required_event.is_set.assert_not_called()
        self.assertEqual(factory.call_args.kwargs,
                         dict(disable_runtime_cache=True, require_engine_cache=True))
