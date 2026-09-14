#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "cyna/poincare.hpp"

namespace cyna {

enum class AxisymTauOrbitStatus : int {
    closed = 0,
    non_axisymmetric_field = 1,
    seed_outside_domain = 2,
    seed_field_nonfinite = 3,
    weak_poloidal_field = 4,
    nontransverse_section = 5,
    nonpositive_radius = 6,
    left_field_domain = 7,
    field_nonfinite = 8,
    return_not_found = 9,
    section_event_failure = 10,
    return_defect_exceeded = 11,
    quadrature_failure = 12,
    directed_section_mismatch = 13,
    tangent_nonfinite = 14,
    tangent_event_failure = 15,
};

inline const char* axisym_tau_orbit_status_name(AxisymTauOrbitStatus status) {
    switch (status) {
    case AxisymTauOrbitStatus::closed:
        return "closed";
    case AxisymTauOrbitStatus::non_axisymmetric_field:
        return "non_axisymmetric_field";
    case AxisymTauOrbitStatus::seed_outside_domain:
        return "seed_outside_domain";
    case AxisymTauOrbitStatus::seed_field_nonfinite:
        return "seed_field_nonfinite";
    case AxisymTauOrbitStatus::weak_poloidal_field:
        return "weak_poloidal_field";
    case AxisymTauOrbitStatus::nontransverse_section:
        return "nontransverse_section";
    case AxisymTauOrbitStatus::nonpositive_radius:
        return "nonpositive_radius";
    case AxisymTauOrbitStatus::left_field_domain:
        return "left_field_domain";
    case AxisymTauOrbitStatus::field_nonfinite:
        return "field_nonfinite";
    case AxisymTauOrbitStatus::return_not_found:
        return "return_not_found";
    case AxisymTauOrbitStatus::section_event_failure:
        return "section_event_failure";
    case AxisymTauOrbitStatus::return_defect_exceeded:
        return "return_defect_exceeded";
    case AxisymTauOrbitStatus::quadrature_failure:
        return "quadrature_failure";
    case AxisymTauOrbitStatus::directed_section_mismatch:
        return "directed_section_mismatch";
    case AxisymTauOrbitStatus::tangent_nonfinite:
        return "tangent_nonfinite";
    case AxisymTauOrbitStatus::tangent_event_failure:
        return "tangent_event_failure";
    }
    return "unknown";
}

struct AxisymTauState {
    double R = std::numeric_limits<double>::quiet_NaN();
    double Z = std::numeric_limits<double>::quiet_NaN();
    double phi = std::numeric_limits<double>::quiet_NaN();
};

struct AxisymTauFieldValue {
    double BR = std::numeric_limits<double>::quiet_NaN();
    double BZ = std::numeric_limits<double>::quiet_NaN();
    double BPhi = std::numeric_limits<double>::quiet_NaN();
};

struct AxisymTauBExtremumEvent {
    bool certified = false;
    double tau = std::numeric_limits<double>::quiet_NaN();
    AxisymTauState position;
    double B = std::numeric_limits<double>::quiet_NaN();
    double d2_B2_dtau2 = std::numeric_limits<double>::quiet_NaN();
    std::size_t interval_index = 0;
    double interval_fraction = std::numeric_limits<double>::quiet_NaN();
};

struct AxisymTauBExtremumEventJVP {
    double tau_jvp = std::numeric_limits<double>::quiet_NaN();
    AxisymTauState position_jvp;
    double B_jvp = std::numeric_limits<double>::quiet_NaN();
};

struct AxisymTauOrbitResult {
    AxisymTauOrbitStatus status = AxisymTauOrbitStatus::return_not_found;
    std::vector<double> tau;
    std::vector<AxisymTauState> path;
    std::vector<double> weights_B2_dtau;
    double period_tau = std::numeric_limits<double>::quiet_NaN();
    double return_defect = std::numeric_limits<double>::quiet_NaN();
    double gauge_measure_B2_dtau = std::numeric_limits<double>::quiet_NaN();
    double phi_advance = std::numeric_limits<double>::quiet_NaN();
    bool opposite_section_crossed = false;
    int accepted_steps = 0;
    AxisymTauBExtremumEvent B_minimum;
    AxisymTauBExtremumEvent B_maximum;
};

struct AxisymTauOrbitJVPResult {
    AxisymTauOrbitResult orbit;
    std::vector<double> tau_jvp;
    std::vector<AxisymTauState> path_jvp;
    std::vector<double> weights_B2_dtau_jvp;
    double period_tau_jvp = std::numeric_limits<double>::quiet_NaN();
    double gauge_measure_B2_dtau_jvp =
        std::numeric_limits<double>::quiet_NaN();
    double phi_advance_jvp = std::numeric_limits<double>::quiet_NaN();
    AxisymTauState return_displacement_jvp;
    double event_fraction = std::numeric_limits<double>::quiet_NaN();
    double event_fraction_jvp = std::numeric_limits<double>::quiet_NaN();
    int section_direction = 0;
    AxisymTauBExtremumEventJVP B_minimum_jvp;
    AxisymTauBExtremumEventJVP B_maximum_jvp;
};

inline bool axisym_tau_field_is_axisymmetric(
    const double* BR,
    const double* BZ,
    const double* BPhi,
    int nR,
    int nZ,
    int nPhi)
{
    if (nPhi <= 1)
        return true;
    const auto component_is_axisymmetric = [&](const double* component) {
        for (int iR = 0; iR < nR; ++iR) {
            for (int iZ = 0; iZ < nZ; ++iZ) {
                const std::size_t base =
                    (static_cast<std::size_t>(iR) * static_cast<std::size_t>(nZ) +
                     static_cast<std::size_t>(iZ)) *
                    static_cast<std::size_t>(nPhi);
                const double reference = component[base];
                if (!std::isfinite(reference))
                    return false;
                for (int iPhi = 1; iPhi < nPhi; ++iPhi) {
                    const double value = component[base + static_cast<std::size_t>(iPhi)];
                    const double scale = std::max({1.0, std::abs(reference), std::abs(value)});
                    if (!std::isfinite(value) ||
                        std::abs(value - reference) > 64.0 *
                            std::numeric_limits<double>::epsilon() * scale)
                        return false;
                }
            }
        }
        return true;
    };
    return component_is_axisymmetric(BR) &&
           component_is_axisymmetric(BZ) &&
           component_is_axisymmetric(BPhi);
}

namespace detail {

struct AxisymTauFieldView {
    const double* BR = nullptr;
    const double* BZ = nullptr;
    const double* BPhi = nullptr;
    const double* R_grid = nullptr;
    const double* Z_grid = nullptr;
    const double* Phi_grid = nullptr;
    int nR = 0;
    int nZ = 0;
    int nPhi = 0;
    double field_period = 2.0 * M_PI;
};

struct AxisymTauFieldDifferential {
    AxisymTauFieldValue value;
    AxisymTauFieldValue dR;
    AxisymTauFieldValue dZ;
    AxisymTauFieldValue dRZ;
};

inline bool axisym_tau_interp_bilinear_differential(
    const double* data,
    const AxisymTauFieldView& field,
    const AxisymTauState& state,
    double& value,
    double& dR,
    double& dZ,
    double& dRZ)
{
    if (!std::isfinite(state.R) || !std::isfinite(state.Z) ||
        state.R < field.R_grid[0] || state.R > field.R_grid[field.nR - 1] ||
        state.Z < field.Z_grid[0] || state.Z > field.Z_grid[field.nZ - 1] ||
        field.nR < 2 || field.nZ < 2)
        return false;
    const double hR = (field.R_grid[field.nR - 1] - field.R_grid[0]) /
        static_cast<double>(field.nR - 1);
    const double hZ = (field.Z_grid[field.nZ - 1] - field.Z_grid[0]) /
        static_cast<double>(field.nZ - 1);
    if (!(hR > 0.0) || !(hZ > 0.0))
        return false;
    const double r_coordinate = (state.R - field.R_grid[0]) / hR;
    const double z_coordinate = (state.Z - field.Z_grid[0]) / hZ;
    int iR = static_cast<int>(r_coordinate);
    int iZ = static_cast<int>(z_coordinate);
    iR = std::max(0, std::min(iR, field.nR - 2));
    iZ = std::max(0, std::min(iZ, field.nZ - 2));
    const double r = r_coordinate - static_cast<double>(iR);
    const double z = z_coordinate - static_cast<double>(iZ);
    const auto sample = [&](int ir, int iz) {
        return data[(static_cast<std::size_t>(ir) *
                     static_cast<std::size_t>(field.nZ) +
                     static_cast<std::size_t>(iz)) *
                    static_cast<std::size_t>(field.nPhi)];
    };
    const double c00 = sample(iR, iZ);
    const double c01 = sample(iR, iZ + 1);
    const double c10 = sample(iR + 1, iZ);
    const double c11 = sample(iR + 1, iZ + 1);
    if (!std::isfinite(c00) || !std::isfinite(c01) ||
        !std::isfinite(c10) || !std::isfinite(c11))
        return false;
    value = (1.0 - r) * ((1.0 - z) * c00 + z * c01) +
        r * ((1.0 - z) * c10 + z * c11);
    dR = ((1.0 - z) * (c10 - c00) + z * (c11 - c01)) / hR;
    dZ = ((1.0 - r) * (c01 - c00) + r * (c11 - c10)) / hZ;
    dRZ = (c11 - c10 - c01 + c00) / (hR * hZ);
    return std::isfinite(value) && std::isfinite(dR) &&
        std::isfinite(dZ) && std::isfinite(dRZ);
}

inline bool axisym_tau_field_differential(
    const AxisymTauFieldView& field,
    const AxisymTauState& state,
    AxisymTauFieldDifferential& result)
{
    return axisym_tau_interp_bilinear_differential(
               field.BR, field, state, result.value.BR,
               result.dR.BR, result.dZ.BR, result.dRZ.BR) &&
           axisym_tau_interp_bilinear_differential(
               field.BZ, field, state, result.value.BZ,
               result.dR.BZ, result.dZ.BZ, result.dRZ.BZ) &&
           axisym_tau_interp_bilinear_differential(
               field.BPhi, field, state, result.value.BPhi,
               result.dR.BPhi, result.dZ.BPhi, result.dRZ.BPhi);
}

inline bool axisym_tau_inside_domain(
    const AxisymTauFieldView& field,
    const AxisymTauState& state)
{
    return std::isfinite(state.R) && std::isfinite(state.Z) &&
           std::isfinite(state.phi) &&
           state.R >= field.R_grid[0] &&
           state.R <= field.R_grid[field.nR - 1] &&
           state.Z >= field.Z_grid[0] &&
           state.Z <= field.Z_grid[field.nZ - 1];
}

inline bool axisym_tau_evaluate_field(
    const AxisymTauFieldView& field,
    const AxisymTauState& state,
    AxisymTauFieldValue& value,
    AxisymTauOrbitStatus& failure)
{
    if (!(state.R > 0.0)) {
        failure = AxisymTauOrbitStatus::nonpositive_radius;
        return false;
    }
    if (!axisym_tau_inside_domain(field, state)) {
        failure = AxisymTauOrbitStatus::left_field_domain;
        return false;
    }
    value.BR = interp3d(
        field.BR,
        field.R_grid,
        field.nR,
        field.Z_grid,
        field.nZ,
        field.Phi_grid,
        field.nPhi,
        state.R,
        state.Z,
        state.phi,
        field.field_period);
    value.BZ = interp3d(
        field.BZ,
        field.R_grid,
        field.nR,
        field.Z_grid,
        field.nZ,
        field.Phi_grid,
        field.nPhi,
        state.R,
        state.Z,
        state.phi,
        field.field_period);
    value.BPhi = interp3d(
        field.BPhi,
        field.R_grid,
        field.nR,
        field.Z_grid,
        field.nZ,
        field.Phi_grid,
        field.nPhi,
        state.R,
        state.Z,
        state.phi,
        field.field_period);
    if (!std::isfinite(value.BR) || !std::isfinite(value.BZ) ||
        !std::isfinite(value.BPhi)) {
        failure = AxisymTauOrbitStatus::field_nonfinite;
        return false;
    }
    return true;
}

inline bool axisym_tau_rhs(
    const AxisymTauFieldView& field,
    const AxisymTauState& state,
    AxisymTauState& rhs,
    AxisymTauOrbitStatus& failure)
{
    AxisymTauFieldValue value;
    if (!axisym_tau_evaluate_field(field, state, value, failure))
        return false;

    // Cylindrical component order is (R, Z, Phi).  Tau is the physical
    // field-line parameter: no division by the poloidal-field magnitude.
    rhs = {value.BR, value.BZ, value.BPhi / state.R};
    if (!std::isfinite(rhs.R) || !std::isfinite(rhs.Z) ||
        !std::isfinite(rhs.phi)) {
        failure = AxisymTauOrbitStatus::field_nonfinite;
        return false;
    }
    return true;
}

inline bool axisym_tau_evaluate_field_jvp(
    const AxisymTauFieldView& field,
    const AxisymTauFieldView& direction,
    const AxisymTauState& state,
    const AxisymTauState& state_jvp,
    AxisymTauFieldValue& value,
    AxisymTauFieldValue& value_jvp,
    AxisymTauOrbitStatus& failure)
{
    if (!(state.R > 0.0)) {
        failure = AxisymTauOrbitStatus::nonpositive_radius;
        return false;
    }
    if (!axisym_tau_inside_domain(field, state)) {
        failure = AxisymTauOrbitStatus::left_field_domain;
        return false;
    }

    double BR_R, BR_Z, BZ_R, BZ_Z, BPhi_R, BPhi_Z;
    if (!interp3d_grad(
            value.BR, BR_R, BR_Z,
            field.BR, field.R_grid, field.nR, field.Z_grid, field.nZ,
            field.Phi_grid, field.nPhi, state.R, state.Z, state.phi,
            field.field_period) ||
        !interp3d_grad(
            value.BZ, BZ_R, BZ_Z,
            field.BZ, field.R_grid, field.nR, field.Z_grid, field.nZ,
            field.Phi_grid, field.nPhi, state.R, state.Z, state.phi,
            field.field_period) ||
        !interp3d_grad(
            value.BPhi, BPhi_R, BPhi_Z,
            field.BPhi, field.R_grid, field.nR, field.Z_grid, field.nZ,
            field.Phi_grid, field.nPhi, state.R, state.Z, state.phi,
            field.field_period)) {
        failure = AxisymTauOrbitStatus::field_nonfinite;
        return false;
    }

    const double delta_BR = interp3d(
        direction.BR, direction.R_grid, direction.nR,
        direction.Z_grid, direction.nZ, direction.Phi_grid, direction.nPhi,
        state.R, state.Z, state.phi, direction.field_period);
    const double delta_BZ = interp3d(
        direction.BZ, direction.R_grid, direction.nR,
        direction.Z_grid, direction.nZ, direction.Phi_grid, direction.nPhi,
        state.R, state.Z, state.phi, direction.field_period);
    const double delta_BPhi = interp3d(
        direction.BPhi, direction.R_grid, direction.nR,
        direction.Z_grid, direction.nZ, direction.Phi_grid, direction.nPhi,
        state.R, state.Z, state.phi, direction.field_period);
    value_jvp = {
        BR_R * state_jvp.R + BR_Z * state_jvp.Z + delta_BR,
        BZ_R * state_jvp.R + BZ_Z * state_jvp.Z + delta_BZ,
        BPhi_R * state_jvp.R + BPhi_Z * state_jvp.Z + delta_BPhi,
    };
    if (!std::isfinite(value_jvp.BR) || !std::isfinite(value_jvp.BZ) ||
        !std::isfinite(value_jvp.BPhi)) {
        failure = AxisymTauOrbitStatus::tangent_nonfinite;
        return false;
    }
    return true;
}

inline bool axisym_tau_rhs_jvp(
    const AxisymTauFieldView& field,
    const AxisymTauFieldView& direction,
    const AxisymTauState& state,
    const AxisymTauState& state_jvp,
    AxisymTauState& rhs,
    AxisymTauState& rhs_jvp,
    AxisymTauOrbitStatus& failure)
{
    AxisymTauFieldValue value, value_jvp;
    if (!axisym_tau_evaluate_field_jvp(
            field, direction, state, state_jvp, value, value_jvp, failure))
        return false;
    rhs = {value.BR, value.BZ, value.BPhi / state.R};
    rhs_jvp = {
        value_jvp.BR,
        value_jvp.BZ,
        value_jvp.BPhi / state.R -
            value.BPhi * state_jvp.R / (state.R * state.R),
    };
    if (!std::isfinite(rhs.R) || !std::isfinite(rhs.Z) ||
        !std::isfinite(rhs.phi) || !std::isfinite(rhs_jvp.R) ||
        !std::isfinite(rhs_jvp.Z) || !std::isfinite(rhs_jvp.phi)) {
        failure = AxisymTauOrbitStatus::tangent_nonfinite;
        return false;
    }
    return true;
}

inline AxisymTauState axisym_tau_axpy(
    const AxisymTauState& state,
    double scale,
    const AxisymTauState& increment)
{
    return {
        state.R + scale * increment.R,
        state.Z + scale * increment.Z,
        state.phi + scale * increment.phi,
    };
}

inline bool axisym_tau_rk4_step(
    const AxisymTauFieldView& field,
    const AxisymTauState& state,
    double step_tau,
    AxisymTauState& next,
    AxisymTauOrbitStatus& failure)
{
    AxisymTauState k1, k2, k3, k4;
    if (!axisym_tau_rhs(field, state, k1, failure) ||
        !axisym_tau_rhs(
            field, axisym_tau_axpy(state, 0.5 * step_tau, k1), k2, failure) ||
        !axisym_tau_rhs(
            field, axisym_tau_axpy(state, 0.5 * step_tau, k2), k3, failure) ||
        !axisym_tau_rhs(
            field, axisym_tau_axpy(state, step_tau, k3), k4, failure))
        return false;

    next = {
        state.R + (step_tau / 6.0) *
            (k1.R + 2.0 * k2.R + 2.0 * k3.R + k4.R),
        state.Z + (step_tau / 6.0) *
            (k1.Z + 2.0 * k2.Z + 2.0 * k3.Z + k4.Z),
        state.phi + (step_tau / 6.0) *
            (k1.phi + 2.0 * k2.phi + 2.0 * k3.phi + k4.phi),
    };
    if (!std::isfinite(next.R) || !std::isfinite(next.Z) ||
        !std::isfinite(next.phi)) {
        failure = AxisymTauOrbitStatus::field_nonfinite;
        return false;
    }
    return true;
}

inline bool axisym_tau_rk4_step_jvp(
    const AxisymTauFieldView& field,
    const AxisymTauFieldView& direction,
    const AxisymTauState& state,
    const AxisymTauState& state_jvp,
    double step_tau,
    AxisymTauState& next,
    AxisymTauState& next_jvp,
    AxisymTauOrbitStatus& failure)
{
    AxisymTauState k1, k2, k3, k4;
    AxisymTauState d1, d2, d3, d4;
    if (!axisym_tau_rhs_jvp(
            field, direction, state, state_jvp, k1, d1, failure) ||
        !axisym_tau_rhs_jvp(
            field, direction,
            axisym_tau_axpy(state, 0.5 * step_tau, k1),
            axisym_tau_axpy(state_jvp, 0.5 * step_tau, d1),
            k2, d2, failure) ||
        !axisym_tau_rhs_jvp(
            field, direction,
            axisym_tau_axpy(state, 0.5 * step_tau, k2),
            axisym_tau_axpy(state_jvp, 0.5 * step_tau, d2),
            k3, d3, failure) ||
        !axisym_tau_rhs_jvp(
            field, direction,
            axisym_tau_axpy(state, step_tau, k3),
            axisym_tau_axpy(state_jvp, step_tau, d3),
            k4, d4, failure))
        return false;

    const double scale = step_tau / 6.0;
    next = {
        state.R + scale * (k1.R + 2.0 * k2.R + 2.0 * k3.R + k4.R),
        state.Z + scale * (k1.Z + 2.0 * k2.Z + 2.0 * k3.Z + k4.Z),
        state.phi + scale *
            (k1.phi + 2.0 * k2.phi + 2.0 * k3.phi + k4.phi),
    };
    next_jvp = {
        state_jvp.R + scale * (d1.R + 2.0 * d2.R + 2.0 * d3.R + d4.R),
        state_jvp.Z + scale * (d1.Z + 2.0 * d2.Z + 2.0 * d3.Z + d4.Z),
        state_jvp.phi + scale *
            (d1.phi + 2.0 * d2.phi + 2.0 * d3.phi + d4.phi),
    };
    if (!std::isfinite(next.R) || !std::isfinite(next.Z) ||
        !std::isfinite(next.phi) || !std::isfinite(next_jvp.R) ||
        !std::isfinite(next_jvp.Z) || !std::isfinite(next_jvp.phi)) {
        failure = AxisymTauOrbitStatus::tangent_nonfinite;
        return false;
    }
    return true;
}

inline AxisymTauState axisym_tau_hermite_state(
    const AxisymTauState& old_state,
    const AxisymTauState& new_state,
    const AxisymTauState& old_rhs,
    const AxisymTauState& new_rhs,
    double step_tau,
    double fraction)
{
    const double value2 = fraction * fraction;
    const double value3 = value2 * fraction;
    const double h00 = 2.0 * value3 - 3.0 * value2 + 1.0;
    const double h10 = value3 - 2.0 * value2 + fraction;
    const double h01 = -2.0 * value3 + 3.0 * value2;
    const double h11 = value3 - value2;
    return {
        h00 * old_state.R + h10 * step_tau * old_rhs.R +
            h01 * new_state.R + h11 * step_tau * new_rhs.R,
        h00 * old_state.Z + h10 * step_tau * old_rhs.Z +
            h01 * new_state.Z + h11 * step_tau * new_rhs.Z,
        h00 * old_state.phi + h10 * step_tau * old_rhs.phi +
            h01 * new_state.phi + h11 * step_tau * new_rhs.phi,
    };
}

inline AxisymTauState axisym_tau_hermite_fraction_derivative(
    const AxisymTauState& old_state,
    const AxisymTauState& new_state,
    const AxisymTauState& old_rhs,
    const AxisymTauState& new_rhs,
    double step_tau,
    double fraction)
{
    const double value2 = fraction * fraction;
    const double h00 = 6.0 * value2 - 6.0 * fraction;
    const double h10 = 3.0 * value2 - 4.0 * fraction + 1.0;
    const double h01 = -6.0 * value2 + 6.0 * fraction;
    const double h11 = 3.0 * value2 - 2.0 * fraction;
    return {
        h00 * old_state.R + h10 * step_tau * old_rhs.R +
            h01 * new_state.R + h11 * step_tau * new_rhs.R,
        h00 * old_state.Z + h10 * step_tau * old_rhs.Z +
            h01 * new_state.Z + h11 * step_tau * new_rhs.Z,
        h00 * old_state.phi + h10 * step_tau * old_rhs.phi +
            h01 * new_state.phi + h11 * step_tau * new_rhs.phi,
    };
}

inline bool axisym_tau_return_event_jvp(
    const AxisymTauFieldView& field,
    const AxisymTauFieldView& direction,
    const AxisymTauState& old_state,
    const AxisymTauState& new_state,
    const AxisymTauState& old_state_jvp,
    const AxisymTauState& new_state_jvp,
    double step_tau,
    double fraction,
    double section_Z_jvp,
    AxisymTauState& fixed_fraction_jvp,
    AxisymTauState& fraction_derivative,
    double& fraction_jvp,
    AxisymTauOrbitStatus& failure)
{
    AxisymTauState old_rhs, new_rhs, old_rhs_jvp, new_rhs_jvp;
    if (!axisym_tau_rhs_jvp(
            field, direction, old_state, old_state_jvp,
            old_rhs, old_rhs_jvp, failure) ||
        !axisym_tau_rhs_jvp(
            field, direction, new_state, new_state_jvp,
            new_rhs, new_rhs_jvp, failure))
        return false;

    fixed_fraction_jvp = axisym_tau_hermite_state(
        old_state_jvp,
        new_state_jvp,
        old_rhs_jvp,
        new_rhs_jvp,
        step_tau,
        fraction);
    fraction_derivative = axisym_tau_hermite_fraction_derivative(
        old_state,
        new_state,
        old_rhs,
        new_rhs,
        step_tau,
        fraction);
    const double scale = std::max({
        std::abs(old_state.Z), std::abs(new_state.Z), 1.0});
    if (!std::isfinite(fraction_derivative.Z) ||
        std::abs(fraction_derivative.Z) <=
            128.0 * std::numeric_limits<double>::epsilon() * scale) {
        failure = AxisymTauOrbitStatus::tangent_event_failure;
        return false;
    }
    fraction_jvp =
        (section_Z_jvp - fixed_fraction_jvp.Z) / fraction_derivative.Z;
    if (!std::isfinite(fraction_jvp)) {
        failure = AxisymTauOrbitStatus::tangent_event_failure;
        return false;
    }
    return true;
}

inline bool axisym_tau_locate_return_event(
    const AxisymTauFieldView& field,
    const AxisymTauState& old_state,
    const AxisymTauState& new_state,
    double step_tau,
    double section_Z,
    int section_direction,
    AxisymTauState& crossing,
    double& fraction,
    AxisymTauOrbitStatus& failure)
{
    AxisymTauState old_rhs, new_rhs;
    if (!axisym_tau_rhs(field, old_state, old_rhs, failure) ||
        !axisym_tau_rhs(field, new_state, new_rhs, failure))
        return false;

    double lower = 0.0;
    double upper = 1.0;
    const double lower_value =
        static_cast<double>(section_direction) * (old_state.Z - section_Z);
    const double upper_value =
        static_cast<double>(section_direction) * (new_state.Z - section_Z);
    if (!(lower_value < 0.0 && upper_value >= 0.0)) {
        failure = AxisymTauOrbitStatus::section_event_failure;
        return false;
    }
    for (int iteration = 0; iteration < 64; ++iteration) {
        const double middle = 0.5 * (lower + upper);
        const AxisymTauState candidate = axisym_tau_hermite_state(
            old_state,
            new_state,
            old_rhs,
            new_rhs,
            step_tau,
            middle);
        const double value =
            static_cast<double>(section_direction) * (candidate.Z - section_Z);
        if (!std::isfinite(value)) {
            failure = AxisymTauOrbitStatus::section_event_failure;
            return false;
        }
        if (value >= 0.0)
            upper = middle;
        else
            lower = middle;
    }
    fraction = 0.5 * (lower + upper);
    crossing = axisym_tau_hermite_state(
        old_state,
        new_state,
        old_rhs,
        new_rhs,
        step_tau,
        fraction);
    crossing.Z = section_Z;
    if (!(fraction > 0.0) || !(fraction <= 1.0) ||
        !std::isfinite(crossing.R) || !std::isfinite(crossing.phi)) {
        failure = AxisymTauOrbitStatus::section_event_failure;
        return false;
    }
    return true;
}

inline bool axisym_tau_B_event_quantities(
    const AxisymTauFieldView& field,
    const AxisymTauState& state,
    double& B2,
    double& dB2_dtau,
    double& d2B2_dtau2)
{
    AxisymTauFieldDifferential f;
    if (!axisym_tau_field_differential(field, state, f))
        return false;
    const double values[3] = {f.value.BR, f.value.BZ, f.value.BPhi};
    const double gradients_R[3] = {f.dR.BR, f.dR.BZ, f.dR.BPhi};
    const double gradients_Z[3] = {f.dZ.BR, f.dZ.BZ, f.dZ.BPhi};
    const double mixed[3] = {f.dRZ.BR, f.dRZ.BZ, f.dRZ.BPhi};
    B2 = 0.0;
    double gradient_B2_R = 0.0;
    double gradient_B2_Z = 0.0;
    double hessian_RR = 0.0;
    double hessian_RZ = 0.0;
    double hessian_ZZ = 0.0;
    for (int component = 0; component < 3; ++component) {
        B2 += values[component] * values[component];
        gradient_B2_R += 2.0 * values[component] * gradients_R[component];
        gradient_B2_Z += 2.0 * values[component] * gradients_Z[component];
        hessian_RR += 2.0 * gradients_R[component] * gradients_R[component];
        hessian_RZ += 2.0 * (
            gradients_R[component] * gradients_Z[component] +
            values[component] * mixed[component]);
        hessian_ZZ += 2.0 * gradients_Z[component] * gradients_Z[component];
    }
    const double vR = f.value.BR;
    const double vZ = f.value.BZ;
    const double acceleration_R = f.dR.BR * vR + f.dZ.BR * vZ;
    const double acceleration_Z = f.dR.BZ * vR + f.dZ.BZ * vZ;
    dB2_dtau = gradient_B2_R * vR + gradient_B2_Z * vZ;
    d2B2_dtau2 = hessian_RR * vR * vR +
        2.0 * hessian_RZ * vR * vZ + hessian_ZZ * vZ * vZ +
        gradient_B2_R * acceleration_R +
        gradient_B2_Z * acceleration_Z;
    return B2 > 0.0 && std::isfinite(B2) && std::isfinite(dB2_dtau) &&
        std::isfinite(d2B2_dtau2);
}

inline bool axisym_tau_B_event_fixed_time_jvp(
    const AxisymTauFieldView& field,
    const AxisymTauFieldView& direction,
    const AxisymTauState& state,
    const AxisymTauState& state_jvp,
    double& B_jvp,
    double& dB2_dtau_jvp)
{
    AxisymTauFieldDifferential f, df;
    if (!axisym_tau_field_differential(field, state, f) ||
        !axisym_tau_field_differential(direction, state, df))
        return false;
    const double values[3] = {f.value.BR, f.value.BZ, f.value.BPhi};
    const double gradients_R[3] = {f.dR.BR, f.dR.BZ, f.dR.BPhi};
    const double gradients_Z[3] = {f.dZ.BR, f.dZ.BZ, f.dZ.BPhi};
    const double mixed[3] = {f.dRZ.BR, f.dRZ.BZ, f.dRZ.BPhi};
    const double direction_values[3] = {
        df.value.BR, df.value.BZ, df.value.BPhi};
    const double direction_gradients_R[3] = {
        df.dR.BR, df.dR.BZ, df.dR.BPhi};
    const double direction_gradients_Z[3] = {
        df.dZ.BR, df.dZ.BZ, df.dZ.BPhi};
    double B2 = 0.0;
    double B2_jvp = 0.0;
    double gradient_B2_R = 0.0;
    double gradient_B2_Z = 0.0;
    double gradient_B2_R_jvp = 0.0;
    double gradient_B2_Z_jvp = 0.0;
    double value_jvp[3];
    for (int component = 0; component < 3; ++component) {
        value_jvp[component] = direction_values[component] +
            gradients_R[component] * state_jvp.R +
            gradients_Z[component] * state_jvp.Z;
        const double gradient_R_jvp = direction_gradients_R[component] +
            mixed[component] * state_jvp.Z;
        const double gradient_Z_jvp = direction_gradients_Z[component] +
            mixed[component] * state_jvp.R;
        B2 += values[component] * values[component];
        B2_jvp += 2.0 * values[component] * value_jvp[component];
        gradient_B2_R += 2.0 * values[component] * gradients_R[component];
        gradient_B2_Z += 2.0 * values[component] * gradients_Z[component];
        gradient_B2_R_jvp += 2.0 * (
            value_jvp[component] * gradients_R[component] +
            values[component] * gradient_R_jvp);
        gradient_B2_Z_jvp += 2.0 * (
            value_jvp[component] * gradients_Z[component] +
            values[component] * gradient_Z_jvp);
    }
    B_jvp = B2_jvp / (2.0 * std::sqrt(B2));
    dB2_dtau_jvp =
        gradient_B2_R_jvp * f.value.BR +
        gradient_B2_Z_jvp * f.value.BZ +
        gradient_B2_R * value_jvp[0] +
        gradient_B2_Z * value_jvp[1];
    return B2 > 0.0 && std::isfinite(B_jvp) &&
        std::isfinite(dB2_dtau_jvp);
}

inline void axisym_tau_find_B_extrema(
    const AxisymTauFieldView& field,
    const AxisymTauOrbitResult& orbit,
    AxisymTauBExtremumEvent& minimum,
    AxisymTauBExtremumEvent& maximum)
{
    if (orbit.path.size() < 3)
        return;
    const double epsilon = std::sqrt(std::numeric_limits<double>::epsilon());
    for (std::size_t index = 0; index + 1 < orbit.path.size(); ++index) {
        AxisymTauState left_rhs, right_rhs;
        AxisymTauOrbitStatus failure = AxisymTauOrbitStatus::field_nonfinite;
        if (!axisym_tau_rhs(field, orbit.path[index], left_rhs, failure) ||
            !axisym_tau_rhs(field, orbit.path[index + 1], right_rhs, failure))
            continue;
        double left_B2, left_g, left_curvature;
        double right_B2, right_g, right_curvature;
        if (!axisym_tau_B_event_quantities(
                field, orbit.path[index], left_B2, left_g, left_curvature) ||
            !axisym_tau_B_event_quantities(
                field, orbit.path[index + 1], right_B2, right_g, right_curvature))
            continue;
        const bool minimum_crossing = left_g < 0.0 && right_g > 0.0;
        const bool maximum_crossing = left_g > 0.0 && right_g < 0.0;
        if (!minimum_crossing && !maximum_crossing)
            continue;
        double lower = 0.0;
        double upper = 1.0;
        double lower_g = left_g;
        AxisymTauState event_state;
        double event_B2 = std::numeric_limits<double>::quiet_NaN();
        double event_g = std::numeric_limits<double>::quiet_NaN();
        double event_curvature = std::numeric_limits<double>::quiet_NaN();
        const double interval = orbit.tau[index + 1] - orbit.tau[index];
        for (int iteration = 0; iteration < 60; ++iteration) {
            const double fraction = 0.5 * (lower + upper);
            event_state = axisym_tau_hermite_state(
                orbit.path[index], orbit.path[index + 1],
                left_rhs, right_rhs, interval, fraction);
            if (!axisym_tau_B_event_quantities(
                    field, event_state, event_B2, event_g, event_curvature))
                break;
            if ((lower_g < 0.0 && event_g < 0.0) ||
                (lower_g > 0.0 && event_g > 0.0)) {
                lower = fraction;
                lower_g = event_g;
            } else {
                upper = fraction;
            }
        }
        const double fraction = 0.5 * (lower + upper);
        const double seam_tolerance =
            256.0 * std::numeric_limits<double>::epsilon();
        if (fraction <= seam_tolerance || fraction >= 1.0 - seam_tolerance)
            continue;
        event_state = axisym_tau_hermite_state(
            orbit.path[index], orbit.path[index + 1],
            left_rhs, right_rhs, interval, fraction);
        if (!axisym_tau_B_event_quantities(
                field, event_state, event_B2, event_g, event_curvature))
            continue;
        const double root_scale = std::max({
            std::abs(left_g), std::abs(right_g), 1.0});
        if (std::abs(event_g) >
            256.0 * std::numeric_limits<double>::epsilon() * root_scale)
            continue;
        const double curvature_scale = std::max(
            std::abs(event_B2) / (interval * interval), 1.0);
        if (std::abs(event_curvature) <= epsilon * curvature_scale)
            continue;
        if ((minimum_crossing && event_curvature <= 0.0) ||
            (maximum_crossing && event_curvature >= 0.0))
            continue;
        AxisymTauBExtremumEvent candidate;
        candidate.certified = true;
        candidate.tau = orbit.tau[index] + fraction * interval;
        candidate.position = event_state;
        candidate.B = std::sqrt(event_B2);
        candidate.d2_B2_dtau2 = event_curvature;
        candidate.interval_index = index;
        candidate.interval_fraction = fraction;
        AxisymTauBExtremumEvent& destination =
            minimum_crossing ? minimum : maximum;
        if (!destination.certified ||
            (minimum_crossing ? candidate.B < destination.B
                              : candidate.B > destination.B))
            destination = candidate;
    }
}

inline bool axisym_tau_B_extremum_jvp(
    const AxisymTauFieldView& field,
    const AxisymTauFieldView& direction,
    const AxisymTauOrbitResult& orbit,
    const std::vector<AxisymTauState>& fixed_tau_jvp,
    const AxisymTauBExtremumEvent& event,
    AxisymTauBExtremumEventJVP& result,
    AxisymTauOrbitStatus& failure)
{
    if (!event.certified || event.interval_index + 1 >= orbit.path.size())
        return false;
    const std::size_t index = event.interval_index;
    const double interval = orbit.tau[index + 1] - orbit.tau[index];
    AxisymTauState left_rhs, right_rhs, left_rhs_jvp, right_rhs_jvp;
    if (!axisym_tau_rhs_jvp(
            field, direction, orbit.path[index], fixed_tau_jvp[index],
            left_rhs, left_rhs_jvp, failure) ||
        !axisym_tau_rhs_jvp(
            field, direction, orbit.path[index + 1], fixed_tau_jvp[index + 1],
            right_rhs, right_rhs_jvp, failure))
        return false;
    const AxisymTauState fixed_event_jvp = axisym_tau_hermite_state(
        fixed_tau_jvp[index], fixed_tau_jvp[index + 1],
        left_rhs_jvp, right_rhs_jvp, interval, event.interval_fraction);
    double fixed_B_jvp, fixed_g_jvp;
    if (!axisym_tau_B_event_fixed_time_jvp(
            field, direction, event.position, fixed_event_jvp,
            fixed_B_jvp, fixed_g_jvp)) {
        failure = AxisymTauOrbitStatus::tangent_event_failure;
        return false;
    }
    result.tau_jvp = -fixed_g_jvp / event.d2_B2_dtau2;
    const AxisymTauState fraction_derivative =
        axisym_tau_hermite_fraction_derivative(
            orbit.path[index], orbit.path[index + 1],
            left_rhs, right_rhs, interval, event.interval_fraction);
    result.position_jvp = axisym_tau_axpy(
        fixed_event_jvp, result.tau_jvp / interval, fraction_derivative);
    result.B_jvp = fixed_B_jvp;
    return std::isfinite(result.tau_jvp) &&
        std::isfinite(result.position_jvp.R) &&
        std::isfinite(result.position_jvp.Z) &&
        std::isfinite(result.position_jvp.phi) &&
        std::isfinite(result.B_jvp);
}

inline void axisym_tau_clear_path(AxisymTauOrbitResult& result) {
    result.tau.clear();
    result.path.clear();
    result.weights_B2_dtau.clear();
}

inline void axisym_tau_clear_jvp(AxisymTauOrbitJVPResult& result) {
    axisym_tau_clear_path(result.orbit);
    result.tau_jvp.clear();
    result.path_jvp.clear();
    result.weights_B2_dtau_jvp.clear();
}

} // namespace detail

inline AxisymTauOrbitResult trace_axisym_tau_closed_orbit(
    double seed_R,
    double section_Z,
    double phi_start,
    double step_tau,
    double maximum_tau,
    double closure_tolerance,
    double minimum_seed_poloidal_field,
    const double* BR,
    const double* BZ,
    const double* BPhi,
    const double* R_grid,
    int nR,
    const double* Z_grid,
    int nZ,
    const double* Phi_grid,
    int nPhi,
    int nfp)
{
    AxisymTauOrbitResult result;
    if (!axisym_tau_field_is_axisymmetric(BR, BZ, BPhi, nR, nZ, nPhi)) {
        result.status = AxisymTauOrbitStatus::non_axisymmetric_field;
        return result;
    }

    const detail::AxisymTauFieldView field{
        BR,
        BZ,
        BPhi,
        R_grid,
        Z_grid,
        Phi_grid,
        nR,
        nZ,
        nPhi,
        2.0 * M_PI / static_cast<double>(nfp),
    };
    AxisymTauState state{seed_R, section_Z, phi_start};
    if (!detail::axisym_tau_inside_domain(field, state)) {
        result.status = AxisymTauOrbitStatus::seed_outside_domain;
        return result;
    }
    if (!(state.R > 0.0)) {
        result.status = AxisymTauOrbitStatus::nonpositive_radius;
        return result;
    }

    AxisymTauFieldValue seed_field;
    AxisymTauOrbitStatus failure = AxisymTauOrbitStatus::seed_field_nonfinite;
    if (!detail::axisym_tau_evaluate_field(field, state, seed_field, failure)) {
        result.status = failure == AxisymTauOrbitStatus::field_nonfinite
            ? AxisymTauOrbitStatus::seed_field_nonfinite
            : failure;
        return result;
    }
    const double seed_Bp = std::hypot(seed_field.BR, seed_field.BZ);
    if (!(seed_Bp > minimum_seed_poloidal_field)) {
        result.status = AxisymTauOrbitStatus::weak_poloidal_field;
        return result;
    }
    if (!(std::abs(seed_field.BZ) > minimum_seed_poloidal_field)) {
        result.status = AxisymTauOrbitStatus::nontransverse_section;
        return result;
    }
    const int section_direction = seed_field.BZ > 0.0 ? 1 : -1;

    const std::size_t maximum_steps = static_cast<std::size_t>(
        std::ceil(maximum_tau / step_tau));
    result.tau.reserve(maximum_steps + 1);
    result.path.reserve(maximum_steps + 1);
    std::vector<double> B2;
    B2.reserve(maximum_steps + 1);
    result.tau.push_back(0.0);
    result.path.push_back(state);
    B2.push_back(
        seed_field.BR * seed_field.BR +
        seed_field.BZ * seed_field.BZ +
        seed_field.BPhi * seed_field.BPhi);

    double tau = 0.0;
    bool returned = false;
    while (tau < maximum_tau &&
           static_cast<std::size_t>(result.accepted_steps) < maximum_steps) {
        const double accepted_step = std::min(step_tau, maximum_tau - tau);
        if (!(accepted_step > 0.0))
            break;
        const AxisymTauState old_state = state;
        AxisymTauState new_state;
        failure = AxisymTauOrbitStatus::field_nonfinite;
        if (!detail::axisym_tau_rk4_step(
                field, old_state, accepted_step, new_state, failure)) {
            result.status = failure;
            detail::axisym_tau_clear_path(result);
            return result;
        }
        if (!detail::axisym_tau_inside_domain(field, new_state)) {
            result.status = AxisymTauOrbitStatus::left_field_domain;
            detail::axisym_tau_clear_path(result);
            return result;
        }

        const double old_signed = static_cast<double>(section_direction) *
            (old_state.Z - section_Z);
        const double new_signed = static_cast<double>(section_direction) *
            (new_state.Z - section_Z);
        if (!result.opposite_section_crossed &&
            old_signed > 0.0 && new_signed <= 0.0)
            result.opposite_section_crossed = true;
        const bool returning = result.opposite_section_crossed &&
            old_signed < 0.0 && new_signed >= 0.0;

        AxisymTauState accepted_state = new_state;
        double effective_step = accepted_step;
        if (returning) {
            double fraction = std::numeric_limits<double>::quiet_NaN();
            if (!detail::axisym_tau_locate_return_event(
                    field,
                    old_state,
                    new_state,
                    accepted_step,
                    section_Z,
                    section_direction,
                    accepted_state,
                    fraction,
                    failure)) {
                result.status = failure;
                detail::axisym_tau_clear_path(result);
                return result;
            }
            effective_step = fraction * accepted_step;
            returned = true;
        }

        AxisymTauFieldValue accepted_field;
        failure = AxisymTauOrbitStatus::field_nonfinite;
        if (!detail::axisym_tau_evaluate_field(
                field, accepted_state, accepted_field, failure)) {
            result.status = failure;
            detail::axisym_tau_clear_path(result);
            return result;
        }
        tau += effective_step;
        if (!(tau > result.tau.back())) {
            result.status = AxisymTauOrbitStatus::section_event_failure;
            detail::axisym_tau_clear_path(result);
            return result;
        }
        state = accepted_state;
        result.tau.push_back(tau);
        result.path.push_back(state);
        B2.push_back(
            accepted_field.BR * accepted_field.BR +
            accepted_field.BZ * accepted_field.BZ +
            accepted_field.BPhi * accepted_field.BPhi);
        ++result.accepted_steps;
        if (returned)
            break;
    }

    if (!returned) {
        result.status = AxisymTauOrbitStatus::return_not_found;
        detail::axisym_tau_clear_path(result);
        return result;
    }

    result.period_tau = result.tau.back();
    result.return_defect = std::hypot(state.R - seed_R, state.Z - section_Z);
    result.phi_advance = state.phi - phi_start;
    if (!std::isfinite(result.return_defect) ||
        result.return_defect > closure_tolerance) {
        result.status = AxisymTauOrbitStatus::return_defect_exceeded;
        detail::axisym_tau_clear_path(result);
        return result;
    }

    result.weights_B2_dtau.assign(result.tau.size(), 0.0);
    double measure = 0.0;
    for (std::size_t index = 0; index + 1 < result.tau.size(); ++index) {
        const double interval = result.tau[index + 1] - result.tau[index];
        const double left_weight = 0.5 * interval * B2[index];
        const double right_weight = 0.5 * interval * B2[index + 1];
        if (!(interval > 0.0) || !std::isfinite(left_weight) ||
            !std::isfinite(right_weight) || left_weight < 0.0 ||
            right_weight < 0.0) {
            result.status = AxisymTauOrbitStatus::quadrature_failure;
            detail::axisym_tau_clear_path(result);
            return result;
        }
        result.weights_B2_dtau[index] += left_weight;
        result.weights_B2_dtau[index + 1] += right_weight;
        measure += left_weight + right_weight;
    }
    if (!(measure > 0.0) || !std::isfinite(measure) ||
        !std::isfinite(result.period_tau) || !(result.period_tau > 0.0) ||
        !std::isfinite(result.phi_advance)) {
        result.status = AxisymTauOrbitStatus::quadrature_failure;
        detail::axisym_tau_clear_path(result);
        return result;
    }
    result.gauge_measure_B2_dtau = measure;
    detail::axisym_tau_find_B_extrema(
        field, result, result.B_minimum, result.B_maximum);
    result.status = AxisymTauOrbitStatus::closed;
    return result;
}

inline AxisymTauOrbitJVPResult trace_axisym_tau_closed_orbit_jvp(
    double seed_R,
    double section_Z,
    double phi_start,
    double seed_R_jvp,
    double section_Z_jvp,
    double phi_start_jvp,
    int expected_section_direction,
    double step_tau,
    double maximum_tau,
    double closure_tolerance,
    double minimum_seed_poloidal_field,
    const double* BR,
    const double* BZ,
    const double* BPhi,
    const double* delta_BR,
    const double* delta_BZ,
    const double* delta_BPhi,
    const double* R_grid,
    int nR,
    const double* Z_grid,
    int nZ,
    const double* Phi_grid,
    int nPhi,
    int nfp)
{
    AxisymTauOrbitJVPResult result;
    auto& orbit = result.orbit;
    if (!axisym_tau_field_is_axisymmetric(BR, BZ, BPhi, nR, nZ, nPhi) ||
        !axisym_tau_field_is_axisymmetric(
            delta_BR, delta_BZ, delta_BPhi, nR, nZ, nPhi)) {
        orbit.status = AxisymTauOrbitStatus::non_axisymmetric_field;
        return result;
    }

    const double field_period = 2.0 * M_PI / static_cast<double>(nfp);
    const detail::AxisymTauFieldView field{
        BR, BZ, BPhi,
        R_grid, Z_grid, Phi_grid, nR, nZ, nPhi, field_period};
    const detail::AxisymTauFieldView direction{
        delta_BR, delta_BZ, delta_BPhi,
        R_grid, Z_grid, Phi_grid, nR, nZ, nPhi, field_period};
    AxisymTauState state{seed_R, section_Z, phi_start};
    AxisymTauState state_jvp{seed_R_jvp, section_Z_jvp, phi_start_jvp};
    if (!detail::axisym_tau_inside_domain(field, state)) {
        orbit.status = AxisymTauOrbitStatus::seed_outside_domain;
        return result;
    }
    if (!(state.R > 0.0)) {
        orbit.status = AxisymTauOrbitStatus::nonpositive_radius;
        return result;
    }

    AxisymTauFieldValue seed_field;
    AxisymTauOrbitStatus failure = AxisymTauOrbitStatus::seed_field_nonfinite;
    if (!detail::axisym_tau_evaluate_field(field, state, seed_field, failure)) {
        orbit.status = failure == AxisymTauOrbitStatus::field_nonfinite
            ? AxisymTauOrbitStatus::seed_field_nonfinite
            : failure;
        return result;
    }
    const double seed_Bp = std::hypot(seed_field.BR, seed_field.BZ);
    if (!(seed_Bp > minimum_seed_poloidal_field)) {
        orbit.status = AxisymTauOrbitStatus::weak_poloidal_field;
        return result;
    }
    if (!(std::abs(seed_field.BZ) > minimum_seed_poloidal_field)) {
        orbit.status = AxisymTauOrbitStatus::nontransverse_section;
        return result;
    }
    const int section_direction = seed_field.BZ > 0.0 ? 1 : -1;
    result.section_direction = section_direction;
    if (section_direction != expected_section_direction) {
        orbit.status = AxisymTauOrbitStatus::directed_section_mismatch;
        return result;
    }

    AxisymTauState seed_rhs, seed_rhs_jvp;
    if (!detail::axisym_tau_rhs_jvp(
            field, direction, state, state_jvp,
            seed_rhs, seed_rhs_jvp, failure)) {
        orbit.status = failure;
        return result;
    }

    const std::size_t maximum_steps = static_cast<std::size_t>(
        std::ceil(maximum_tau / step_tau));
    orbit.tau.reserve(maximum_steps + 1);
    orbit.path.reserve(maximum_steps + 1);
    std::vector<double> B2;
    std::vector<AxisymTauState> fixed_tau_jvp;
    std::vector<AxisymTauState> time_derivative;
    B2.reserve(maximum_steps + 1);
    fixed_tau_jvp.reserve(maximum_steps + 1);
    time_derivative.reserve(maximum_steps + 1);
    orbit.tau.push_back(0.0);
    orbit.path.push_back(state);
    B2.push_back(
        seed_field.BR * seed_field.BR +
        seed_field.BZ * seed_field.BZ +
        seed_field.BPhi * seed_field.BPhi);
    fixed_tau_jvp.push_back(state_jvp);
    time_derivative.push_back(seed_rhs);

    double tau = 0.0;
    bool returned = false;
    while (tau < maximum_tau &&
           static_cast<std::size_t>(orbit.accepted_steps) < maximum_steps) {
        const double accepted_step = std::min(step_tau, maximum_tau - tau);
        if (!(accepted_step > 0.0))
            break;
        const AxisymTauState old_state = state;
        const AxisymTauState old_state_jvp = state_jvp;
        AxisymTauState new_state, new_state_jvp;
        failure = AxisymTauOrbitStatus::field_nonfinite;
        if (!detail::axisym_tau_rk4_step_jvp(
                field, direction, old_state, old_state_jvp, accepted_step,
                new_state, new_state_jvp, failure)) {
            orbit.status = failure;
            detail::axisym_tau_clear_jvp(result);
            return result;
        }
        if (!detail::axisym_tau_inside_domain(field, new_state)) {
            orbit.status = AxisymTauOrbitStatus::left_field_domain;
            detail::axisym_tau_clear_jvp(result);
            return result;
        }

        const double old_signed = static_cast<double>(section_direction) *
            (old_state.Z - section_Z);
        const double new_signed = static_cast<double>(section_direction) *
            (new_state.Z - section_Z);
        if (!orbit.opposite_section_crossed &&
            old_signed > 0.0 && new_signed <= 0.0)
            orbit.opposite_section_crossed = true;
        const bool returning = orbit.opposite_section_crossed &&
            old_signed < 0.0 && new_signed >= 0.0;

        AxisymTauState accepted_state = new_state;
        AxisymTauState accepted_fixed_jvp = new_state_jvp;
        AxisymTauState accepted_time_derivative;
        AxisymTauState ignored_rhs_jvp;
        double effective_step = accepted_step;
        if (returning) {
            double fraction = std::numeric_limits<double>::quiet_NaN();
            if (!detail::axisym_tau_locate_return_event(
                    field, old_state, new_state, accepted_step, section_Z,
                    section_direction, accepted_state, fraction, failure)) {
                orbit.status = failure;
                detail::axisym_tau_clear_jvp(result);
                return result;
            }
            AxisymTauState fraction_derivative;
            double fraction_jvp = std::numeric_limits<double>::quiet_NaN();
            if (!detail::axisym_tau_return_event_jvp(
                    field, direction, old_state, new_state,
                    old_state_jvp, new_state_jvp, accepted_step, fraction,
                    section_Z_jvp, accepted_fixed_jvp, fraction_derivative,
                    fraction_jvp, failure)) {
                orbit.status = failure;
                detail::axisym_tau_clear_jvp(result);
                return result;
            }
            accepted_time_derivative = {
                fraction_derivative.R / accepted_step,
                fraction_derivative.Z / accepted_step,
                fraction_derivative.phi / accepted_step,
            };
            effective_step = fraction * accepted_step;
            result.period_tau_jvp = fraction_jvp * accepted_step;
            result.event_fraction = fraction;
            result.event_fraction_jvp = fraction_jvp;
            returned = true;
        } else if (!detail::axisym_tau_rhs_jvp(
                       field, direction, new_state, new_state_jvp,
                       accepted_time_derivative, ignored_rhs_jvp, failure)) {
            orbit.status = failure;
            detail::axisym_tau_clear_jvp(result);
            return result;
        }

        AxisymTauFieldValue accepted_field;
        failure = AxisymTauOrbitStatus::field_nonfinite;
        if (!detail::axisym_tau_evaluate_field(
                field, accepted_state, accepted_field, failure)) {
            orbit.status = failure;
            detail::axisym_tau_clear_jvp(result);
            return result;
        }
        tau += effective_step;
        if (!(tau > orbit.tau.back())) {
            orbit.status = AxisymTauOrbitStatus::section_event_failure;
            detail::axisym_tau_clear_jvp(result);
            return result;
        }
        state = accepted_state;
        state_jvp = accepted_fixed_jvp;
        orbit.tau.push_back(tau);
        orbit.path.push_back(state);
        B2.push_back(
            accepted_field.BR * accepted_field.BR +
            accepted_field.BZ * accepted_field.BZ +
            accepted_field.BPhi * accepted_field.BPhi);
        fixed_tau_jvp.push_back(accepted_fixed_jvp);
        time_derivative.push_back(accepted_time_derivative);
        ++orbit.accepted_steps;
        if (returned)
            break;
    }

    if (!returned) {
        orbit.status = AxisymTauOrbitStatus::return_not_found;
        detail::axisym_tau_clear_jvp(result);
        return result;
    }

    orbit.period_tau = orbit.tau.back();
    orbit.return_defect = std::hypot(state.R - seed_R, state.Z - section_Z);
    orbit.phi_advance = state.phi - phi_start;
    if (!std::isfinite(orbit.return_defect) ||
        orbit.return_defect > closure_tolerance) {
        orbit.status = AxisymTauOrbitStatus::return_defect_exceeded;
        detail::axisym_tau_clear_jvp(result);
        return result;
    }

    const std::size_t node_count = orbit.tau.size();
    result.tau_jvp.assign(node_count, 0.0);
    result.path_jvp.resize(node_count);
    result.weights_B2_dtau_jvp.assign(node_count, 0.0);
    std::vector<double> B2_jvp(node_count, 0.0);
    for (std::size_t index = 0; index < node_count; ++index) {
        const double phase = orbit.tau[index] / orbit.period_tau;
        result.tau_jvp[index] = phase * result.period_tau_jvp;
        result.path_jvp[index] = detail::axisym_tau_axpy(
            fixed_tau_jvp[index],
            result.tau_jvp[index],
            time_derivative[index]);
        AxisymTauFieldValue value, value_jvp;
        failure = AxisymTauOrbitStatus::tangent_nonfinite;
        if (!detail::axisym_tau_evaluate_field_jvp(
                field, direction, orbit.path[index], result.path_jvp[index],
                value, value_jvp, failure)) {
            orbit.status = failure;
            detail::axisym_tau_clear_jvp(result);
            return result;
        }
        B2_jvp[index] = 2.0 * (
            value.BR * value_jvp.BR +
            value.BZ * value_jvp.BZ +
            value.BPhi * value_jvp.BPhi);
        if (!std::isfinite(B2_jvp[index])) {
            orbit.status = AxisymTauOrbitStatus::tangent_nonfinite;
            detail::axisym_tau_clear_jvp(result);
            return result;
        }
    }
    result.path_jvp.back().Z = section_Z_jvp;

    orbit.weights_B2_dtau.assign(node_count, 0.0);
    double measure = 0.0;
    double measure_jvp = 0.0;
    for (std::size_t index = 0; index + 1 < node_count; ++index) {
        const double interval = orbit.tau[index + 1] - orbit.tau[index];
        const double interval_jvp =
            result.tau_jvp[index + 1] - result.tau_jvp[index];
        const double left_weight = 0.5 * interval * B2[index];
        const double right_weight = 0.5 * interval * B2[index + 1];
        const double left_weight_jvp = 0.5 * (
            interval_jvp * B2[index] + interval * B2_jvp[index]);
        const double right_weight_jvp = 0.5 * (
            interval_jvp * B2[index + 1] + interval * B2_jvp[index + 1]);
        if (!(interval > 0.0) || !std::isfinite(left_weight) ||
            !std::isfinite(right_weight) || left_weight < 0.0 ||
            right_weight < 0.0 || !std::isfinite(left_weight_jvp) ||
            !std::isfinite(right_weight_jvp)) {
            orbit.status = AxisymTauOrbitStatus::quadrature_failure;
            detail::axisym_tau_clear_jvp(result);
            return result;
        }
        orbit.weights_B2_dtau[index] += left_weight;
        orbit.weights_B2_dtau[index + 1] += right_weight;
        result.weights_B2_dtau_jvp[index] += left_weight_jvp;
        result.weights_B2_dtau_jvp[index + 1] += right_weight_jvp;
        measure += left_weight + right_weight;
        measure_jvp += left_weight_jvp + right_weight_jvp;
    }
    if (!(measure > 0.0) || !std::isfinite(measure) ||
        !std::isfinite(measure_jvp) || !(orbit.period_tau > 0.0) ||
        !std::isfinite(result.period_tau_jvp)) {
        orbit.status = AxisymTauOrbitStatus::quadrature_failure;
        detail::axisym_tau_clear_jvp(result);
        return result;
    }
    orbit.gauge_measure_B2_dtau = measure;
    result.gauge_measure_B2_dtau_jvp = measure_jvp;
    result.phi_advance_jvp =
        result.path_jvp.back().phi - phi_start_jvp;
    result.return_displacement_jvp = {
        result.path_jvp.back().R - seed_R_jvp,
        result.path_jvp.back().Z - section_Z_jvp,
        result.phi_advance_jvp,
    };
    detail::axisym_tau_find_B_extrema(
        field, orbit, orbit.B_minimum, orbit.B_maximum);
    AxisymTauOrbitStatus extrema_failure = AxisymTauOrbitStatus::tangent_event_failure;
    if (orbit.B_minimum.certified)
        detail::axisym_tau_B_extremum_jvp(
            field, direction, orbit, fixed_tau_jvp, orbit.B_minimum,
            result.B_minimum_jvp, extrema_failure);
    extrema_failure = AxisymTauOrbitStatus::tangent_event_failure;
    if (orbit.B_maximum.certified)
        detail::axisym_tau_B_extremum_jvp(
            field, direction, orbit, fixed_tau_jvp, orbit.B_maximum,
            result.B_maximum_jvp, extrema_failure);
    orbit.status = AxisymTauOrbitStatus::closed;
    return result;
}

} // namespace cyna
