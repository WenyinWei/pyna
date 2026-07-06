from pathlib import Path

import numpy as np
import pytest

from pyna.benchmark.vmec_counterparts import (
    CounterpartResult,
    VmecBenchmarkReport,
    compare_counterpart_results,
    write_vmec_benchmark_outputs,
)


def _reader(name, *, transpose=False, scalar_delta=0.0):
    rmnc = np.arange(12, dtype=float).reshape(3, 4)
    arrays = {
        "iotaf": np.array([0.8, 0.9, 1.0]),
        "presf": np.array([3.0, 2.0, 1.0]),
        "phi": np.array([0.0, 0.5, 1.0]),
        "rmnc": rmnc.T if transpose else rmnc,
        "zmns": (rmnc + 1.0).T if transpose else rmnc + 1.0,
        "bmnc": (rmnc + 2.0).T if transpose else rmnc + 2.0,
    }
    summary = {
        "scalars": {"ns": 3, "mnmax": 4, "mnmax_nyq": 4, "betatotal": 0.01 + scalar_delta},
        "profiles": {},
        "modes": {},
    }
    return CounterpartResult(name=name, status="ok", runtime_s=0.01, summary=summary, arrays=arrays)


def test_compare_counterpart_results_normalises_vmec_mode_orientation():
    readers = {"netcdf": _reader("netcdf"), "simsopt": _reader("simsopt", transpose=True)}

    comparisons = compare_counterpart_results(readers)

    assert comparisons["status"] == "ok"
    modes = comparisons["readers"]["simsopt"]["modes"]
    assert modes["rmnc"]["status"] == "ok"
    assert modes["rmnc"]["max_abs_error"] == 0.0


def test_compare_counterpart_results_reports_scalar_errors():
    readers = {"netcdf": _reader("netcdf"), "vmecpp": _reader("vmecpp", scalar_delta=0.001)}

    comparisons = compare_counterpart_results(readers)

    beta = comparisons["readers"]["vmecpp"]["scalars"]["betatotal"]
    assert beta["status"] == "ok"
    assert beta["abs_error"] == pytest.approx(0.001)


def test_write_vmec_benchmark_outputs_omits_arrays_from_json(tmp_path):
    report = VmecBenchmarkReport(
        source={"basename": "wout_public.nc", "sha256_16": "abc", "size_bytes": 123},
        readers={"netcdf": _reader("netcdf")},
        comparisons={"status": "ok", "readers": {}},
    )

    paths = write_vmec_benchmark_outputs(report, tmp_path, make_plots=False)

    text = Path(paths["json"]).read_text(encoding="utf-8")
    assert "local_path" not in text
    assert "arrays" not in text
    assert Path(paths["scalar_csv"]).exists()
