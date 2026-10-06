"""WorkerPool shuts down by finishing or skipping tasks, never by killing busy workers.

``multiprocessing.Pool.terminate()`` can kill a worker while it holds the result queue's lock,
after which ``terminate()`` itself blocks forever. These tests check that leaving a
:class:`~relucent._internal.parallel.WorkerPool` early lets every worker exit normally
(exit code 0) rather than by SIGTERM.
"""

import time

import pytest

from relucent._internal.parallel import BlockingQueue, worker_pool


def _slow_square(x: int) -> int:
    time.sleep(0.01)
    return x * x


def _raise_on_three(x: int) -> int:
    if x == 3:
        raise ValueError("three")
    return x


_initialized_value: int | None = None


def _set_value(value: int) -> None:
    global _initialized_value
    _initialized_value = value


def _read_value(_: int) -> int | None:
    return _initialized_value


def _worker_processes(pool):
    return list(pool._pool._pool)


def test_map_starmap_and_initializer():
    with worker_pool(2) as pool:
        assert pool.map(_slow_square, range(5)) == [0, 1, 4, 9, 16]
        assert pool.starmap(pow, [(2, 3), (3, 2)]) == [8, 9]
    with worker_pool(2, initializer=_set_value, initargs=(7,)) as pool:
        assert pool.map(_read_value, range(4)) == [7, 7, 7, 7]


@pytest.mark.parametrize("chunksize", [1, 7])
def test_imap_matches_serial(chunksize: int):
    with worker_pool(2) as pool:
        assert list(pool.imap(_slow_square, range(20), chunksize=chunksize)) == [x * x for x in range(20)]
        assert sorted(pool.imap_unordered(_slow_square, range(20), chunksize=chunksize)) == [x * x for x in range(20)]


@pytest.mark.parametrize("chunksize", [1, 7])
def test_early_break_skips_remaining_tasks_and_workers_exit_normally(chunksize: int):
    start = time.perf_counter()
    with worker_pool(4) as pool:
        results = pool.imap_unordered(_slow_square, range(100_000), chunksize=chunksize)
        first = next(results)
        workers = _worker_processes(pool)
    elapsed = time.perf_counter() - start
    assert first in {x * x for x in range(1_000)}
    assert all(p.exitcode == 0 for p in workers), [p.exitcode for p in workers]
    # Running every task would take about 100_000 * 0.01 / 4 = 250 s.
    assert elapsed < 60


def test_blocking_queue_input_closed_before_exit():
    queue = BlockingQueue()
    for x in range(200):
        queue.push(x)
    with worker_pool(2) as pool:
        try:
            for i, _ in enumerate(pool.imap_unordered(_slow_square, queue)):
                if i == 5:
                    break
        finally:
            queue.close()
        workers = _worker_processes(pool)
    assert all(p.exitcode == 0 for p in workers), [p.exitcode for p in workers]


def test_task_errors_do_not_block_shutdown():
    with worker_pool(2) as pool:
        pool.imap_unordered(_raise_on_three, range(20))  # never consumed, including the failing task
        workers = _worker_processes(pool)
    assert all(p.exitcode == 0 for p in workers), [p.exitcode for p in workers]


def test_error_in_body_propagates_after_clean_shutdown():
    workers = []
    with pytest.raises(RuntimeError, match="stop"), worker_pool(2) as pool:
        next(pool.imap_unordered(_slow_square, range(1_000)))
        workers = _worker_processes(pool)
        raise RuntimeError("stop")
    assert workers and all(p.exitcode == 0 for p in workers), [p.exitcode for p in workers]


def test_keyboard_interrupt_terminates_at_once():
    workers = []
    with pytest.raises(KeyboardInterrupt), worker_pool(2) as pool:
        pool.imap_unordered(_slow_square, range(100_000))
        workers = _worker_processes(pool)
        raise KeyboardInterrupt
    assert workers and all(p.exitcode is not None for p in workers)


def test_shutdown_is_idempotent():
    pool = worker_pool(2)
    assert pool.map(_slow_square, [3]) == [9]
    pool.shutdown()
    pool.shutdown()
