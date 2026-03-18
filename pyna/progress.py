"""Progress reporting for long-running parallel traces.

This module provides a pluggable, back-end-agnostic progress-reporting
abstraction that can be attached to any batch computation in pyna (field-line
tracing, connection-length maps, topology scans, etc.).

Design
------
All reporters share the same interface defined by :class:`TraceProgressBase`.
The caller creates one reporter and passes it to the computation:

    tracer.trace_many(starts, t_max, progress=TqdmProgress())

Reporters are composable via :class:`CompositeProgress` so that, e.g., a
tqdm bar and a log-file writer can run simultaneously:

    prog = CompositeProgress([TqdmProgress(), LogFileProgress("run.jsonl")])
    tracer.trace_many(starts, t_max, progress=prog)

Available reporters
-------------------
* :class:`NullProgress`     — silent no-op (default when ``progress=None``).
* :class:`TqdmProgress`     — tqdm progress bar; auto-detects Jupyter
  notebooks and switches to ``tqdm.notebook`` automatically; supports both
  *per-task* and *aggregate* (total-count) modes.
* :class:`LogFileProgress`  — appends JSON-Lines records to a file so that
  an external watcher can poll it to detect stale runs.
* :class:`CompositeProgress`— fan-out to multiple reporters.

Staleness detection
-------------------
:class:`LogFileProgress` writes a heartbeat record on every ``update()``
call.  An external monitor can read the file and compare the ``"timestamp"``
field against wall-clock time to decide whether a run has stalled.

Thread safety
-------------
All reporters are designed to be called from multiple threads simultaneously.
Internal locking is minimal but sufficient for the increment operations.
"""
from __future__ import annotations

import abc
import json
import os
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class TraceProgressBase(abc.ABC):
    """Abstract base for all progress reporters.

    Sub-classes must implement :meth:`start`, :meth:`update`, and
    :meth:`close`.  The :meth:`__enter__` / :meth:`__exit__` context
    manager protocol calls ``start`` and ``close`` automatically.
    """

    @abc.abstractmethod
    def start(self, total: int, description: str = "") -> None:
        """Initialise reporting for a batch of *total* tasks.

        Parameters
        ----------
        total : int
            Number of tasks in the batch.
        description : str
            Short human-readable label for the batch.
        """

    @abc.abstractmethod
    def update(
        self,
        task_id: int,
        *,
        steps_done: int = 0,
        steps_total: int = 0,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Report progress on a single task.

        Parameters
        ----------
        task_id : int
            Index of the task (0-based).
        steps_done : int
            Number of integration steps completed so far for this task.
        steps_total : int
            Total number of integration steps planned for this task.
            ``0`` means the total is unknown.
        info : dict or None
            Arbitrary extra metadata (e.g. current position, arc length).
        """

    @abc.abstractmethod
    def close(self) -> None:
        """Finalise and flush all output."""

    # ------------------------------------------------------------------
    # Context manager helpers
    # ------------------------------------------------------------------

    def __enter__(self) -> "TraceProgressBase":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# NullProgress — silent default
# ---------------------------------------------------------------------------

class NullProgress(TraceProgressBase):
    """No-op progress reporter.  Zero overhead, zero output.

    This is the default when no ``progress=`` argument is supplied.
    """

    def start(self, total: int, description: str = "") -> None:
        pass

    def update(self, task_id: int, *, steps_done: int = 0,
               steps_total: int = 0, info: Optional[Dict[str, Any]] = None) -> None:
        pass

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# TqdmProgress — interactive progress bar
# ---------------------------------------------------------------------------

class TqdmProgress(TraceProgressBase):
    """Progress bar backed by `tqdm`.

    Automatically switches between ``tqdm.notebook`` (Jupyter) and
    ``tqdm.auto`` (CLI / other environments).

    Parameters
    ----------
    per_task : bool
        ``True``  — show one progress bar per task (stacked bars in notebooks,
                    separate lines in CLI).  Good for a small number of long
                    tasks.
        ``False`` — show a single aggregate bar counting completed tasks.
                    Good for many short tasks.
    description : str
        Default description used for the aggregate bar (or prefix for
        per-task bars).
    leave : bool
        Passed to tqdm: whether to leave the bar visible after completion.
    ncols : int or None
        Terminal width hint for tqdm.
    colour : str or None
        Bar colour string accepted by tqdm (e.g. ``"green"``, ``"#ff0000"``).

    Raises
    ------
    ImportError
        If tqdm is not installed.  Install with ``pip install tqdm`` or
        ``pip install pyna-chaos[dev]``.

    Examples
    --------
    Aggregate mode (default) — one bar counting finished traces::

        with TqdmProgress() as prog:
            tracer.trace_many(starts, t_max, progress=prog)

    Per-task mode — one bar per field line::

        with TqdmProgress(per_task=True) as prog:
            tracer.trace_many(starts, t_max, progress=prog)
    """

    def __init__(
        self,
        per_task: bool = False,
        description: str = "tracing",
        leave: bool = True,
        ncols: Optional[int] = None,
        colour: Optional[str] = None,
    ) -> None:
        try:
            from tqdm.auto import tqdm as _tqdm  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "TqdmProgress requires tqdm.  "
                "Install it with: pip install tqdm"
            ) from exc

        self.per_task = per_task
        self.description = description
        self.leave = leave
        self.ncols = ncols
        self.colour = colour

        self._total: int = 0
        self._agg_bar: Any = None            # aggregate bar
        self._task_bars: Dict[int, Any] = {} # per-task bars
        self._lock = threading.Lock()

    def _make_bar(self, total: int, desc: str) -> Any:
        from tqdm.auto import tqdm
        return tqdm(
            total=total,
            desc=desc,
            leave=self.leave,
            ncols=self.ncols,
            colour=self.colour,
        )

    def start(self, total: int, description: str = "") -> None:
        self._total = total
        desc = description or self.description
        if not self.per_task:
            self._agg_bar = self._make_bar(total, desc)

    def update(
        self,
        task_id: int,
        *,
        steps_done: int = 0,
        steps_total: int = 0,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._lock:
            if self.per_task:
                self._update_per_task(task_id, steps_done, steps_total)
            else:
                # Aggregate: count when a task is marked complete
                # Convention: steps_done == steps_total (and both > 0)
                # or steps_done == -1 signals task completion.
                if steps_done < 0 or (steps_total > 0 and steps_done >= steps_total):
                    if self._agg_bar is not None:
                        self._agg_bar.update(1)

    def _update_per_task(self, task_id: int, steps_done: int, steps_total: int) -> None:
        if task_id not in self._task_bars:
            desc = f"{self.description}[{task_id}]"
            bar = self._make_bar(steps_total if steps_total > 0 else 0, desc)
            self._task_bars[task_id] = bar

        bar = self._task_bars[task_id]
        if steps_total > 0 and bar.total != steps_total:
            bar.total = steps_total
            bar.refresh()

        if steps_done < 0:
            # Task complete
            bar.n = bar.total or 0
            bar.refresh()
        else:
            # Set absolute position (tqdm update is incremental)
            delta = steps_done - bar.n
            if delta > 0:
                bar.update(delta)

    def close(self) -> None:
        with self._lock:
            if self._agg_bar is not None:
                self._agg_bar.close()
                self._agg_bar = None
            for bar in self._task_bars.values():
                bar.close()
            self._task_bars.clear()


# ---------------------------------------------------------------------------
# LogFileProgress — JSON-Lines disk writer for staleness detection
# ---------------------------------------------------------------------------

class LogFileProgress(TraceProgressBase):
    """Write JSON-Lines progress records to a file for external monitoring.

    Each :meth:`update` call appends one record to *path* so that an
    external process (monitoring script, dashboard, MPI rank) can poll the
    file and read the ``"timestamp"`` field to determine whether the run has
    stalled.

    Record format (one JSON object per line)::

        {"event": "start",  "total": 10, "description": "tracing",
         "timestamp": 1710000000.123}
        {"event": "update", "task_id": 3, "steps_done": 42,
         "steps_total": 100, "timestamp": 1710000012.456, "info": {...}}
        {"event": "close",  "timestamp": 1710000099.789}

    Parameters
    ----------
    path : str or path-like
        File path to write records to.  The file is created or appended to.
    flush_every : int
        Flush the underlying file descriptor every *flush_every* writes to
        balance I/O overhead with freshness.  Default ``1`` (always flush).

    Examples
    --------
    Monitor staleness from a separate script::

        import json, time
        with open("run.jsonl") as f:
            lines = f.readlines()
        last = json.loads(lines[-1])
        age = time.time() - last["timestamp"]
        print(f"Last update was {age:.0f}s ago")
    """

    def __init__(self, path: Union[str, os.PathLike], flush_every: int = 1) -> None:
        self.path = str(path)
        self.flush_every = max(1, flush_every)
        self._fp = None
        self._lock = threading.Lock()
        self._write_count = 0

    def _write(self, record: Dict[str, Any]) -> None:
        """Append one JSON record to the file (caller holds the lock)."""
        record["timestamp"] = time.time()
        line = json.dumps(record, default=str) + "\n"
        self._fp.write(line)
        self._write_count += 1
        if self._write_count % self.flush_every == 0:
            self._fp.flush()

    def start(self, total: int, description: str = "") -> None:
        with self._lock:
            self._fp = open(self.path, "a", encoding="utf-8")  # noqa: WPS515
            self._write({"event": "start", "total": total,
                         "description": description})

    def update(
        self,
        task_id: int,
        *,
        steps_done: int = 0,
        steps_total: int = 0,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._lock:
            if self._fp is None:
                return
            record: Dict[str, Any] = {
                "event": "update",
                "task_id": task_id,
                "steps_done": steps_done,
                "steps_total": steps_total,
            }
            if info:
                record["info"] = info
            self._write(record)

    def close(self) -> None:
        with self._lock:
            if self._fp is not None:
                self._write({"event": "close"})
                self._fp.close()
                self._fp = None


# ---------------------------------------------------------------------------
# CompositeProgress — fan-out to multiple reporters
# ---------------------------------------------------------------------------

class CompositeProgress(TraceProgressBase):
    """Fan-out progress reporter that delegates to multiple back-ends.

    Parameters
    ----------
    reporters : sequence of TraceProgressBase
        List of reporters to notify on every call.

    Examples
    --------
    Combine a tqdm bar with a log file::

        prog = CompositeProgress([
            TqdmProgress(),
            LogFileProgress("trace_run.jsonl"),
        ])
        with prog:
            tracer.trace_many(starts, t_max, progress=prog)
    """

    def __init__(self, reporters: Sequence[TraceProgressBase]) -> None:
        self._reporters = list(reporters)

    def start(self, total: int, description: str = "") -> None:
        for r in self._reporters:
            r.start(total, description)

    def update(
        self,
        task_id: int,
        *,
        steps_done: int = 0,
        steps_total: int = 0,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        for r in self._reporters:
            r.update(task_id, steps_done=steps_done,
                     steps_total=steps_total, info=info)

    def close(self) -> None:
        for r in self._reporters:
            r.close()


# ---------------------------------------------------------------------------
# Helper: normalise the progress= argument
# ---------------------------------------------------------------------------

def _coerce_progress(progress: Optional[TraceProgressBase]) -> TraceProgressBase:
    """Return *progress* unchanged, or a :class:`NullProgress` if ``None``."""
    if progress is None:
        return NullProgress()
    if not isinstance(progress, TraceProgressBase):
        raise TypeError(
            f"progress must be a TraceProgressBase instance or None; "
            f"got {type(progress)!r}"
        )
    return progress
