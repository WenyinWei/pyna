// cyna/bindings/coil_field_grid_bindings.hpp
// ===========================================
// pybind11 registration for cyna::coil_field_grid_cpu.
//
// Included by bindings/flt_bindings.cpp so the public symbol lives in the
// single _cyna_ext module (CPU-only, no CUDA linkage).
#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "cyna/coil_field_grid.hpp"

namespace cyna {
namespace bindings {

namespace py = pybind11;

namespace detail {

inline void require_contiguous_double(
    const py::array& a, const char* name)
{
    if (!(a.flags() & py::array::c_style))
        throw std::runtime_error(std::string(name) + " must be C-contiguous");
    if (a.dtype().num() != py::dtype::of<double>().num())
        throw std::runtime_error(std::string(name) + " must be float64");
}

inline void require_finite_increasing_axis(
    const double* p, py::ssize_t n, const char* name, bool require_positive)
{
    if (n < 1)
        throw std::runtime_error(std::string(name) + " must be non-empty");
    for (py::ssize_t i = 0; i < n; ++i) {
        if (!std::isfinite(p[i]))
            throw std::runtime_error(
                std::string(name) + " must contain only finite values");
        if (require_positive && !(p[i] > 0.0))
            throw std::runtime_error(
                std::string(name) + " must contain only positive values");
        if (i > 0 && !(p[i] > p[i - 1]))
            throw std::runtime_error(
                std::string(name) + " must be strictly increasing");
    }
}

}  // namespace detail

/// Build the cylindrical field grid of a coil set.
///
/// Returns (BR, BZ, BPhi), each a (nR, nZ, nPhi) C-order float64 array in
/// Tesla.  The CPU kernel is OpenMP-parallel over grid rows; the GIL is
/// released while it runs.
inline py::tuple py_coil_field_grid(
    py::array_t<double> segments,
    py::array_t<double> currents,
    py::array_t<double> R_grid,
    py::array_t<double> Z_grid,
    py::array_t<double> Phi_grid,
    int nfp,
    bool phi_closed,
    int n_threads)
{
    detail::require_contiguous_double(segments, "segments");
    detail::require_contiguous_double(currents, "currents");
    detail::require_contiguous_double(R_grid, "R_grid");
    detail::require_contiguous_double(Z_grid, "Z_grid");
    detail::require_contiguous_double(Phi_grid, "Phi_grid");

    if (nfp < 1)
        throw std::runtime_error("nfp must be a positive integer");
    if (segments.ndim() != 3 || segments.shape(2) != 3)
        throw std::runtime_error("segments must have shape (K, S, 3)");
    const py::ssize_t n_coils = segments.shape(0);
    const py::ssize_t n_points = segments.shape(1);
    if (n_coils < 1 || n_points < 2)
        throw std::runtime_error(
            "segments must contain at least one polyline with two points");
    if (currents.ndim() != 1 || currents.size() != n_coils)
        throw std::runtime_error("currents must have shape (K,) matching segments");

    if (R_grid.ndim() != 1 || Z_grid.ndim() != 1 || Phi_grid.ndim() != 1)
        throw std::runtime_error("R_grid, Z_grid and Phi_grid must be 1-D");

    const py::ssize_t nR = R_grid.size();
    const py::ssize_t nZ = Z_grid.size();
    const py::ssize_t nPhi = Phi_grid.size();

    detail::require_finite_increasing_axis(R_grid.data(), nR, "R_grid", true);
    detail::require_finite_increasing_axis(Z_grid.data(), nZ, "Z_grid", false);
    detail::require_finite_increasing_axis(Phi_grid.data(), nPhi, "Phi_grid", false);

    const double period = 2.0 * std::acos(-1.0) / static_cast<double>(nfp);
    const double tol = 1e-12 * std::max(1.0, std::abs(period));
    if (nPhi > 1) {
        const double span = Phi_grid.data()[nPhi - 1] - Phi_grid.data()[0];
        if (phi_closed) {
            if (std::abs(span - period) > tol)
                throw std::runtime_error(
                    "Phi_grid must be a closed periodic grid spanning exactly "
                    "2*pi/nfp when phi_closed=True; nfp is explicit and is "
                    "never inferred from Phi_grid");
        } else if (span > period + tol) {
            throw std::runtime_error(
                "Phi_grid spans more than one field period 2*pi/nfp");
        }
    }

    const std::vector<py::ssize_t> shape{nR, nZ, nPhi};
    py::array_t<double> BR(shape);
    py::array_t<double> BZ(shape);
    py::array_t<double> BPhi(shape);

    double* pBR = BR.mutable_data();
    double* pBZ = BZ.mutable_data();
    double* pBPhi = BPhi.mutable_data();

    {
        py::gil_scoped_release release;
        cyna::coil_field_grid_cpu(
            segments.data(), static_cast<int>(n_coils),
            static_cast<int>(n_points), currents.data(),
            R_grid.data(), static_cast<int>(nR),
            Z_grid.data(), static_cast<int>(nZ),
            Phi_grid.data(), static_cast<int>(nPhi),
            pBR, pBZ, pBPhi, phi_closed, n_threads);
    }
    return py::make_tuple(BR, BZ, BPhi);
}

inline void bind_coil_field_grid(py::module_& m)
{
    m.def(
        "coil_field_grid",
        &py_coil_field_grid,
        py::arg("segments"),
        py::arg("currents"),
        py::arg("R_grid"),
        py::arg("Z_grid"),
        py::arg("Phi_grid"),
        py::arg("nfp") = 1,
        py::arg("phi_closed") = true,
        py::arg("n_threads") = -1,
        "Biot-Savart coil field on a cylindrical (nR, nZ, nPhi) grid.\n\n"
        "segments: (K, S, 3) float64 Cartesian coil polylines (m).\n"
        "currents: (K,) float64 signed ampere-turns.\n"
        "R_grid/Z_grid/Phi_grid: strictly increasing float64 axes.\n"
        "phi_closed=True expects Phi_grid[-1] = Phi_grid[0] + 2*pi/nfp.\n"
        "Returns (BR, BZ, BPhi) in Tesla, each (nR, nZ, nPhi) C-order.\n"
        "CPU-only; uses OpenMP when the extension was built with it.");
}

}  // namespace bindings
}  // namespace cyna
