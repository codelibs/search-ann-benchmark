"""Utilities for reducing measurement noise during latency benchmarks.

The functions and context manager defined here aim to minimize sources of
jitter that can dominate sub-millisecond search latencies:

- Pause Python's cyclic garbage collector during the measurement window so
  GC pauses do not appear inside individual query timings.
- Raise process priority (``os.nice``) so the benchmark loop is less likely
  to be descheduled by background tasks.
- Pin the process to a single CPU core (Linux only) so per-query timings
  do not include cross-core migration costs.

All operations degrade gracefully: failure to elevate priority or set CPU
affinity is logged at WARNING level but does not abort the benchmark.
"""

from __future__ import annotations

import gc
import os
from collections.abc import Iterator
from contextlib import contextmanager

from search_ann_benchmark.core.logging import get_logger

logger = get_logger("measurement")


def _try_raise_priority(nice_delta: int = -10) -> int | None:
    """Attempt to raise the current process priority.

    Args:
        nice_delta: Value passed to ``os.nice``. Negative values increase
            priority and typically require elevated permissions on Linux.

    Returns:
        The new nice value on success, or ``None`` if priority could not be
        adjusted (e.g. ``PermissionError`` on a non-privileged user).
    """
    try:
        new_nice = os.nice(nice_delta)
        logger.debug(f"Raised process priority: nice={new_nice}")
        return new_nice
    except (PermissionError, OSError) as exc:
        logger.warning(
            f"Could not raise process priority via os.nice({nice_delta}): "
            f"{type(exc).__name__}: {exc}. "
            "Run with elevated privileges or set CAP_SYS_NICE for tighter timings."
        )
        return None


def _try_pin_cpu() -> frozenset[int] | None:
    """Pin the current process to a single CPU core.

    Returns:
        The previous CPU affinity mask on success (so it can be restored),
        or ``None`` if affinity could not be applied (non-Linux platform,
        ``PermissionError``, etc.).
    """
    if not hasattr(os, "sched_setaffinity") or not hasattr(os, "sched_getaffinity"):
        logger.debug("CPU affinity not supported on this platform; skipping pin.")
        return None
    try:
        previous = frozenset(os.sched_getaffinity(0))  # type: ignore[attr-defined]
        if not previous:
            logger.warning("Empty CPU affinity mask; skipping pin.")
            return None
        target = min(previous)
        os.sched_setaffinity(0, {target})  # type: ignore[attr-defined]
        logger.debug(f"Pinned process to CPU {target} (previous affinity: {sorted(previous)})")
        return previous
    except (PermissionError, OSError) as exc:
        logger.warning(
            f"Could not set CPU affinity: {type(exc).__name__}: {exc}. "
            "Latency measurements may include cross-core migration jitter."
        )
        return None


def _restore_cpu_affinity(previous: frozenset[int] | None) -> None:
    """Restore CPU affinity to the previously captured mask."""
    if previous is None:
        return
    if not hasattr(os, "sched_setaffinity"):
        return
    try:
        os.sched_setaffinity(0, set(previous))  # type: ignore[attr-defined]
        logger.debug(f"Restored CPU affinity: {sorted(previous)}")
    except OSError as exc:
        logger.warning(f"Could not restore CPU affinity: {type(exc).__name__}: {exc}")


@contextmanager
def low_noise_measurement(
    *,
    pin_cpu: bool = True,
    raise_priority: bool = True,
    disable_gc: bool = True,
) -> Iterator[None]:
    """Context manager that reduces measurement noise during a benchmark.

    On entry the manager:
        1. Runs a full ``gc.collect()`` to drain pending cycles, then
           disables the cyclic GC (if ``disable_gc`` is True).
        2. Raises process priority via ``os.nice`` (if ``raise_priority``
           is True). Failures are logged but otherwise ignored.
        3. Pins the process to a single CPU core (if ``pin_cpu`` is True
           and the platform supports ``os.sched_setaffinity``).

    On exit the GC is re-enabled and CPU affinity is restored to its
    previous mask. ``os.nice`` cannot be lowered back (it is monotonic for
    unprivileged processes) so priority is left as-is.

    Args:
        pin_cpu: Whether to pin to a single CPU core on Linux.
        raise_priority: Whether to attempt to raise process priority.
        disable_gc: Whether to disable the cyclic garbage collector.

    Yields:
        None.
    """
    gc_was_enabled = gc.isenabled()
    previous_affinity: frozenset[int] | None = None
    try:
        if disable_gc:
            gc.collect()
            gc.disable()
            logger.debug("Disabled cyclic GC for measurement window.")
        if raise_priority:
            _try_raise_priority()
        if pin_cpu:
            previous_affinity = _try_pin_cpu()
        yield
    finally:
        if disable_gc and gc_was_enabled:
            gc.enable()
            logger.debug("Re-enabled cyclic GC after measurement window.")
        if pin_cpu:
            _restore_cpu_affinity(previous_affinity)


__all__ = ["low_noise_measurement"]
