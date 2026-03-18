"""Tests for pyna.progress — pluggable progress-reporting abstraction.

Tests cover:
* NullProgress — zero-overhead no-op
* LogFileProgress — JSON-Lines file output and staleness detection
* CompositeProgress — fan-out to multiple reporters
* _coerce_progress — type-checking helper
* Integration with FieldLineTracer.trace_many
* Integration with connection_length
"""
from __future__ import annotations

import json
import os
import tempfile
import time
import threading
from pathlib import Path

import numpy as np
import pytest

from pyna.progress import (
    NullProgress,
    LogFileProgress,
    CompositeProgress,
    TraceProgressBase,
    _coerce_progress,
)
from pyna.flt import FieldLineTracer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pure_toroidal(rzphi):
    R = rzphi[0]
    B_phi = 1.0 / R
    return np.array([0.0, 0.0, B_phi / (R * B_phi)])


STARTS = np.array([[2.0, 0.0, 0.0], [2.5, 0.0, 0.0], [3.0, 0.0, 0.0]])


# ---------------------------------------------------------------------------
# NullProgress
# ---------------------------------------------------------------------------

class TestNullProgress:
    def test_start_does_nothing(self):
        p = NullProgress()
        p.start(5, "test")  # should not raise

    def test_update_does_nothing(self):
        p = NullProgress()
        p.start(3)
        p.update(0, steps_done=10, steps_total=100)  # should not raise

    def test_close_does_nothing(self):
        p = NullProgress()
        p.close()  # should not raise

    def test_context_manager(self):
        with NullProgress() as p:
            p.start(2)
            p.update(0, steps_done=-1)
            p.update(1, steps_done=-1)


# ---------------------------------------------------------------------------
# LogFileProgress
# ---------------------------------------------------------------------------

class TestLogFileProgress:
    def test_creates_file(self, tmp_path):
        path = tmp_path / "progress.jsonl"
        p = LogFileProgress(str(path))
        p.start(3, "flt")
        p.close()
        assert path.exists()

    def test_start_record(self, tmp_path):
        path = tmp_path / "progress.jsonl"
        p = LogFileProgress(str(path))
        p.start(5, "flt_test")
        p.close()

        lines = path.read_text().splitlines()
        start_rec = json.loads(lines[0])
        assert start_rec["event"] == "start"
        assert start_rec["total"] == 5
        assert start_rec["description"] == "flt_test"
        assert "timestamp" in start_rec

    def test_update_record(self, tmp_path):
        path = tmp_path / "progress.jsonl"
        p = LogFileProgress(str(path))
        p.start(2)
        p.update(0, steps_done=50, steps_total=100, info={"R": 1.8})
        p.close()

        lines = path.read_text().splitlines()
        update_rec = json.loads(lines[1])
        assert update_rec["event"] == "update"
        assert update_rec["task_id"] == 0
        assert update_rec["steps_done"] == 50
        assert update_rec["steps_total"] == 100
        assert update_rec["info"]["R"] == 1.8

    def test_close_record(self, tmp_path):
        path = tmp_path / "progress.jsonl"
        p = LogFileProgress(str(path))
        p.start(1)
        p.close()

        lines = path.read_text().splitlines()
        close_rec = json.loads(lines[-1])
        assert close_rec["event"] == "close"

    def test_timestamp_is_recent(self, tmp_path):
        path = tmp_path / "progress.jsonl"
        before = time.time()
        p = LogFileProgress(str(path))
        p.start(1)
        p.close()
        after = time.time()

        lines = path.read_text().splitlines()
        for line in lines:
            rec = json.loads(line)
            assert before <= rec["timestamp"] <= after + 1.0

    def test_appends_on_reopen(self, tmp_path):
        path = tmp_path / "progress.jsonl"
        # First run
        p1 = LogFileProgress(str(path))
        p1.start(2)
        p1.close()
        # Second run appends
        p2 = LogFileProgress(str(path))
        p2.start(3)
        p2.close()

        lines = path.read_text().splitlines()
        assert len(lines) >= 4, "Should have 2 runs worth of records"

    def test_thread_safety(self, tmp_path):
        """Multiple threads calling update() simultaneously must not corrupt output."""
        path = tmp_path / "threaded.jsonl"
        p = LogFileProgress(str(path))
        p.start(20)

        errors = []

        def worker(i):
            try:
                p.update(i, steps_done=i * 10, steps_total=100)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        p.close()

        assert not errors, f"Errors during threaded update: {errors}"
        lines = path.read_text().splitlines()
        for line in lines:
            json.loads(line)  # must be valid JSON

    def test_context_manager(self, tmp_path):
        path = tmp_path / "ctx.jsonl"
        with LogFileProgress(str(path)) as p:
            p.start(1)
            p.update(0, steps_done=-1)
        # File should be closed after context exit
        lines = path.read_text().splitlines()
        close_rec = json.loads(lines[-1])
        assert close_rec["event"] == "close"

    def test_staleness_detection(self, tmp_path):
        """Simulates reading the log file to check staleness."""
        path = tmp_path / "stale.jsonl"
        p = LogFileProgress(str(path))
        p.start(10)
        p.update(0, steps_done=5, steps_total=10)
        p.close()

        lines = path.read_text().splitlines()
        last_update = None
        for line in lines:
            rec = json.loads(line)
            if rec["event"] == "update":
                last_update = rec
        assert last_update is not None
        age = time.time() - last_update["timestamp"]
        assert age < 5.0, f"Recorded update should be recent (age={age:.2f}s)"


# ---------------------------------------------------------------------------
# CompositeProgress
# ---------------------------------------------------------------------------

class TestCompositeProgress:
    def test_fan_out_start(self, tmp_path):
        path = tmp_path / "comp.jsonl"
        a = LogFileProgress(str(path))
        b = NullProgress()
        comp = CompositeProgress([a, b])
        comp.start(5, "composite_test")
        comp.close()

        lines = path.read_text().splitlines()
        assert json.loads(lines[0])["event"] == "start"

    def test_fan_out_update(self, tmp_path):
        path = tmp_path / "comp2.jsonl"
        a = LogFileProgress(str(path))
        b = NullProgress()
        comp = CompositeProgress([a, b])
        comp.start(3)
        comp.update(1, steps_done=20, steps_total=50)
        comp.close()

        lines = path.read_text().splitlines()
        update_lines = [l for l in lines if '"update"' in l]
        assert len(update_lines) == 1
        rec = json.loads(update_lines[0])
        assert rec["task_id"] == 1
        assert rec["steps_done"] == 20

    def test_context_manager(self, tmp_path):
        path = tmp_path / "comp3.jsonl"
        with CompositeProgress([LogFileProgress(str(path)), NullProgress()]) as comp:
            comp.start(1)
            comp.update(0, steps_done=-1)
        lines = path.read_text().splitlines()
        assert any('"close"' in l for l in lines)


# ---------------------------------------------------------------------------
# _coerce_progress helper
# ---------------------------------------------------------------------------

class TestCoerceProgress:
    def test_none_returns_null(self):
        result = _coerce_progress(None)
        assert isinstance(result, NullProgress)

    def test_passes_through_valid(self):
        p = NullProgress()
        assert _coerce_progress(p) is p

    def test_raises_on_invalid_type(self):
        with pytest.raises(TypeError, match="TraceProgressBase"):
            _coerce_progress("not_a_progress")

    def test_raises_on_number(self):
        with pytest.raises(TypeError):
            _coerce_progress(42)


# ---------------------------------------------------------------------------
# Integration: FieldLineTracer.trace_many with progress=
# ---------------------------------------------------------------------------

class TestFieldLineTracerProgress:
    def test_trace_many_null_progress(self):
        """trace_many with NullProgress should not alter results."""
        tracer = FieldLineTracer(_pure_toroidal, dt=0.1)
        results_plain = tracer.trace_many(STARTS, t_max=5.0)
        results_prog  = tracer.trace_many(STARTS, t_max=5.0,
                                          progress=NullProgress())
        assert len(results_plain) == len(results_prog)
        for a, b in zip(results_plain, results_prog):
            np.testing.assert_array_equal(a, b)

    def test_trace_many_log_progress(self, tmp_path):
        """trace_many should write N completion records to the log file."""
        path = tmp_path / "flt_prog.jsonl"
        tracer = FieldLineTracer(_pure_toroidal, dt=0.1)
        with LogFileProgress(str(path)) as prog:
            tracer.trace_many(STARTS, t_max=5.0, progress=prog)

        lines = path.read_text().splitlines()
        update_lines = [l for l in lines if '"update"' in l]
        # One completion record per starting point
        assert len(update_lines) == len(STARTS)
        for line in update_lines:
            rec = json.loads(line)
            assert rec["steps_done"] == -1

    def test_trace_many_composite_progress(self, tmp_path):
        """CompositeProgress routes to all reporters."""
        path = tmp_path / "composite_flt.jsonl"
        tracer = FieldLineTracer(_pure_toroidal, dt=0.1)
        with CompositeProgress([LogFileProgress(str(path)), NullProgress()]) as prog:
            tracer.trace_many(STARTS, t_max=5.0, progress=prog)

        lines = path.read_text().splitlines()
        # Should have: 1 start + N updates + 1 close
        assert len(lines) >= len(STARTS) + 2

    def test_trace_many_results_unchanged_with_progress(self, tmp_path):
        """Results must be identical whether or not progress reporting is active."""
        path = tmp_path / "check.jsonl"
        tracer = FieldLineTracer(_pure_toroidal, dt=0.1)
        baseline = tracer.trace_many(STARTS, t_max=5.0)
        with LogFileProgress(str(path)) as prog:
            with_prog = tracer.trace_many(STARTS, t_max=5.0, progress=prog)
        for a, b in zip(baseline, with_prog):
            np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# Integration: connection_length with progress=
# ---------------------------------------------------------------------------

class TestConnectionLengthProgress:
    """Verify that progress= does not alter connection_length results."""

    def _make_wall(self):
        theta = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        R_wall = 2.0 + 0.5 * np.cos(theta)
        Z_wall = 0.5 * np.sin(theta)
        return R_wall, Z_wall

    def _vert_field(self, R, Z, phi):
        return np.array([0.0, 0.15])

    def test_null_progress_no_change(self):
        from pyna.connection_length import connection_length
        wall = self._make_wall()
        starts = np.array([[2.0, 0.0], [2.0, 0.1]])
        baseline = connection_length(self._vert_field, starts, wall,
                                     direction="+", max_turns=20, dphi=0.05)
        with_prog = connection_length(self._vert_field, starts, wall,
                                      direction="+", max_turns=20, dphi=0.05,
                                      progress=NullProgress())
        np.testing.assert_array_equal(baseline["L_plus"], with_prog["L_plus"])

    def test_log_progress_writes_records(self, tmp_path):
        from pyna.connection_length import connection_length
        path = tmp_path / "cl_prog.jsonl"
        wall = self._make_wall()
        starts = np.array([[2.0, 0.0], [2.0, 0.1]])
        with LogFileProgress(str(path)) as prog:
            connection_length(self._vert_field, starts, wall,
                              direction="+", max_turns=20, dphi=0.05,
                              progress=prog)
        lines = path.read_text().splitlines()
        update_lines = [l for l in lines if '"update"' in l]
        assert len(update_lines) == len(starts)
