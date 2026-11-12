"""Exercise the installed wheel from outside its source directory (no pytest)."""

from importlib.machinery import EXTENSION_SUFFIXES
from importlib.metadata import distribution
from pathlib import Path
import sys

import numpy as np

import pyna
import pyna._cyna as cyna


def main():
    source_root = Path(__file__).resolve().parents[1]
    package_file = Path(pyna.__file__).resolve()
    installed_package = Path(
        distribution("pyna-chaos").locate_file("pyna/__init__.py")
    ).resolve()
    assert not package_file.is_relative_to(source_root), (
        f"Source checkout shadows the installed wheel: {package_file}"
    )
    assert package_file == installed_package, (
        f"pyna does not come from the installed distribution: {package_file}"
    )
    assert cyna.is_available(), "The wheel's native extension did not load"
    assert callable(cyna.coil_field_grid), "The CPU coil-field kernel is missing"
    assert callable(cyna.trace_poincare_batch), "The CPU Poincare tracer is missing"
    extension = cyna._cyna_ext
    extension_file = Path(extension.__file__).resolve()
    assert extension_file.parent == package_file.parent / "_cyna", (
        f"Native extension loaded from outside the installed package: {extension_file}"
    )
    assert any(str(extension_file).endswith(suffix) for suffix in EXTENSION_SUFFIXES)
    assert cyna.get_version() not in ("", "not available")

    # A purely toroidal field leaves R and Z unchanged during a native RK4 step.
    # Reuse the public buffer helper to exercise the installed NumPy bridge too.
    shape = (2, 2, 4)
    field = cyna.prepare_field_cache({
        "R_grid": np.array([1.0, 2.0]),
        "Z_grid": np.array([-1.0, 1.0]),
        "Phi_grid": np.linspace(0.0, 2.0 * np.pi, shape[2], endpoint=False),
        "BR": np.zeros(shape),
        "BZ": np.zeros(shape),
        "BPhi": np.ones(shape),
    })
    result = extension.rk4_step_test(
        1.5, 0.25, 0.1, 0.05,
        **{key: field[key] for key in (
            "BR", "BZ", "BPhi", "R_grid", "Z_grid", "Phi_grid", "nfp"
        )},
    )
    np.testing.assert_allclose(result, (1.5, 0.25), rtol=0.0, atol=1.0e-14)
    print(f"Wheel smoke passed: Python {sys.version.split()[0]}, "
          f"pyna {pyna.__version__}, native {cyna.get_version()}")
    print(f"Package: {package_file}")
    print(f"Extension: {extension_file}")


if __name__ == "__main__":
    main()
