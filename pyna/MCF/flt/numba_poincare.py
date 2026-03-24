"""
High-performance numba-parallel Poincaré field-line tracer.

Coordinate system: cylindrical (R, phi, Z).
Field-line ODE (phi as integration parameter):
    dR/dphi = R * BR / BPhi
    dZ/dphi = R * BZ / BPhi

Poincaré section: record (R, Z) each time phi crosses phi_section (mod 2π).
Wall termination: stop tracing when seed leaves the wall polygon.

All hot-path functions are @numba.njit; batch function uses numba.prange
for automatic parallelism.
"""

import numpy as np
import numba


# ---------------------------------------------------------------------------
# Low-level numba helpers
# ---------------------------------------------------------------------------

@numba.njit(inline='always')
def _bisect(arr, val, n):
    """Return index i such that arr[i] <= val < arr[i+1], clamped to [0, n-2]."""
    lo, hi = 0, n - 1
    while lo < hi - 1:
        mid = (lo + hi) >> 1
        if arr[mid] <= val:
            lo = mid
        else:
            hi = mid
    if lo < 0:
        lo = 0
    if lo > n - 2:
        lo = n - 2
    return lo


@numba.njit(inline='always')
def _interp3d(val_flat, R, Z, Phi,
              R_grid, Z_grid, Phi_grid,
              nx, ny, nz):
    """
    Trilinear interpolation on a regular (R, Z, Phi) grid.
    Array layout: val_flat[iR * ny*nz + iZ * nz + iPhi]
    Phi is periodic on [0, 2pi).
    Out-of-bounds R/Z are clamped to edge values.
    """
    TWO_PI = 6.283185307179586

    # --- Phi periodic wrap ---
    Phi_mod = Phi % TWO_PI
    if Phi_mod < 0.0:
        Phi_mod += TWO_PI

    # --- clamp R, Z ---
    R_c = R
    if R_c < R_grid[0]:
        R_c = R_grid[0]
    elif R_c > R_grid[nx - 1]:
        R_c = R_grid[nx - 1]

    Z_c = Z
    if Z_c < Z_grid[0]:
        Z_c = Z_grid[0]
    elif Z_c > Z_grid[ny - 1]:
        Z_c = Z_grid[ny - 1]

    # --- bracket indices ---
    iR = _bisect(R_grid, R_c, nx)
    iZ = _bisect(Z_grid, Z_c, ny)
    iPhi = _bisect(Phi_grid, Phi_mod, nz)

    # fractional coordinates
    dR = R_grid[iR + 1] - R_grid[iR]
    dZ = Z_grid[iZ + 1] - Z_grid[iZ]
    dPhi = Phi_grid[iPhi + 1] - Phi_grid[iPhi]

    tR = (R_c - R_grid[iR]) / dR if dR != 0.0 else 0.0
    tZ = (Z_c - Z_grid[iZ]) / dZ if dZ != 0.0 else 0.0
    tPhi = (Phi_mod - Phi_grid[iPhi]) / dPhi if dPhi != 0.0 else 0.0

    # handle periodic Phi wrap-around for the upper bracket
    iPhi1 = (iPhi + 1) % nz

    # trilinear (inlined index arithmetic — no closures inside njit)
    s_R = ny * nz
    val = (
        (1 - tR) * (1 - tZ) * (1 - tPhi) * val_flat[iR     * s_R + iZ     * nz + iPhi ] +
        (1 - tR) * (1 - tZ) *      tPhi  * val_flat[iR     * s_R + iZ     * nz + iPhi1] +
        (1 - tR) *      tZ  * (1 - tPhi) * val_flat[iR     * s_R + (iZ+1) * nz + iPhi ] +
        (1 - tR) *      tZ  *      tPhi  * val_flat[iR     * s_R + (iZ+1) * nz + iPhi1] +
             tR  * (1 - tZ) * (1 - tPhi) * val_flat[(iR+1) * s_R + iZ     * nz + iPhi ] +
             tR  * (1 - tZ) *      tPhi  * val_flat[(iR+1) * s_R + iZ     * nz + iPhi1] +
             tR  *      tZ  * (1 - tPhi) * val_flat[(iR+1) * s_R + (iZ+1) * nz + iPhi ] +
             tR  *      tZ  *      tPhi  * val_flat[(iR+1) * s_R + (iZ+1) * nz + iPhi1]
    )
    return val


@numba.njit(inline='always')
def _point_in_polygon(px, py, poly_x, poly_y, n_poly):
    """Ray-casting point-in-polygon test. Returns True if (px,py) is inside."""
    inside = False
    j = n_poly - 1
    for i in range(n_poly):
        xi = poly_x[i]; yi = poly_y[i]
        xj = poly_x[j]; yj = poly_y[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


@numba.njit(inline='always')
def _B_at(R_pt, Z_pt, Phi_pt,
          R_grid, Z_grid, Phi_grid,
          BR_flat, BPhi_flat, BZ_flat,
          nx, ny, nz):
    """Return (BR, BPhi, BZ) at a point."""
    br   = _interp3d(BR_flat,   R_pt, Z_pt, Phi_pt, R_grid, Z_grid, Phi_grid, nx, ny, nz)
    bphi = _interp3d(BPhi_flat, R_pt, Z_pt, Phi_pt, R_grid, Z_grid, Phi_grid, nx, ny, nz)
    bz   = _interp3d(BZ_flat,   R_pt, Z_pt, Phi_pt, R_grid, Z_grid, Phi_grid, nx, ny, nz)
    return br, bphi, bz


@numba.njit(inline='always')
def _rhs(R_pt, Z_pt, Phi_pt,
         R_grid, Z_grid, Phi_grid,
         BR_flat, BPhi_flat, BZ_flat,
         nx, ny, nz):
    """Returns (dR/dphi, dZ/dphi)."""
    br, bphi, bz = _B_at(R_pt, Z_pt, Phi_pt,
                          R_grid, Z_grid, Phi_grid,
                          BR_flat, BPhi_flat, BZ_flat,
                          nx, ny, nz)
    if bphi == 0.0:
        return 0.0, 0.0
    return R_pt * br / bphi, R_pt * bz / bphi


@numba.njit(inline='always')
def _rk4_step(R_pt, Z_pt, Phi_pt, dphi,
              R_grid, Z_grid, Phi_grid,
              BR_flat, BPhi_flat, BZ_flat,
              nx, ny, nz):
    """Single RK4 step. Returns (R_new, Z_new)."""
    k1R, k1Z = _rhs(R_pt, Z_pt, Phi_pt, R_grid, Z_grid, Phi_grid, BR_flat, BPhi_flat, BZ_flat, nx, ny, nz)
    h2 = dphi * 0.5
    k2R, k2Z = _rhs(R_pt + h2 * k1R, Z_pt + h2 * k1Z, Phi_pt + h2,
                    R_grid, Z_grid, Phi_grid, BR_flat, BPhi_flat, BZ_flat, nx, ny, nz)
    k3R, k3Z = _rhs(R_pt + h2 * k2R, Z_pt + h2 * k2Z, Phi_pt + h2,
                    R_grid, Z_grid, Phi_grid, BR_flat, BPhi_flat, BZ_flat, nx, ny, nz)
    k4R, k4Z = _rhs(R_pt + dphi * k3R, Z_pt + dphi * k3Z, Phi_pt + dphi,
                    R_grid, Z_grid, Phi_grid, BR_flat, BPhi_flat, BZ_flat, nx, ny, nz)
    R_new = R_pt + (dphi / 6.0) * (k1R + 2.0 * k2R + 2.0 * k3R + k4R)
    Z_new = Z_pt + (dphi / 6.0) * (k1Z + 2.0 * k2Z + 2.0 * k3Z + k4Z)
    return R_new, Z_new


# ---------------------------------------------------------------------------
# Single-section batch tracer
# ---------------------------------------------------------------------------

@numba.njit(parallel=True)
def _trace_batch_njit(R_seeds, Z_seeds, phi_section, N_turns, DPhi,
                      R_grid, Z_grid, Phi_grid,
                      BR_flat, BPhi_flat, BZ_flat,
                      wall_R, wall_Z,
                      nx, ny, nz):
    """
    Core njit parallel tracer for a single Poincaré section.
    Returns (poi_counts, poi_R_out, poi_Z_out) where out arrays have size
    N_seeds * N_turns (pre-allocated, use counts to slice).
    """
    TWO_PI = 6.283185307179586
    N_seeds = len(R_seeds)
    n_wall = len(wall_R)
    steps_per_turn = int(round(TWO_PI / DPhi))
    max_pts = N_seeds * N_turns

    poi_counts = np.zeros(N_seeds, dtype=np.int64)
    poi_R_out  = np.empty(max_pts, dtype=np.float64)
    poi_Z_out  = np.empty(max_pts, dtype=np.float64)

    # Each seed writes to its own slice: seed s → offset s*N_turns
    for s in numba.prange(N_seeds):
        R_pt = R_seeds[s]
        Z_pt = Z_seeds[s]

        # Check initial wall condition
        if not _point_in_polygon(R_pt, Z_pt, wall_R, wall_Z, n_wall):
            poi_counts[s] = 0
            continue

        offset = s * N_turns
        count = 0
        phi_pt = phi_section  # start at section

        total_steps = steps_per_turn * N_turns
        for step in range(total_steps):
            phi_prev = phi_pt
            R_new, Z_new = _rk4_step(R_pt, Z_pt, phi_pt, DPhi,
                                      R_grid, Z_grid, Phi_grid,
                                      BR_flat, BPhi_flat, BZ_flat,
                                      nx, ny, nz)
            phi_new = phi_prev + DPhi

            # Wall check
            if not _point_in_polygon(R_new, Z_new, wall_R, wall_Z, n_wall):
                break

            # Poincaré crossing: did phi cross phi_section + k*2pi?
            # We detect each time we complete a full turn from start
            # i.e., when (step+1) is a multiple of steps_per_turn
            if (step + 1) % steps_per_turn == 0 and count < N_turns:
                poi_R_out[offset + count] = R_new
                poi_Z_out[offset + count] = Z_new
                count += 1

            R_pt = R_new
            Z_pt = Z_new
            phi_pt = phi_new

        poi_counts[s] = count

    return poi_counts, poi_R_out, poi_Z_out


# ---------------------------------------------------------------------------
# Multi-section batch tracer
# ---------------------------------------------------------------------------

@numba.njit(parallel=True)
def _trace_multi_njit(R_seeds, Z_seeds, phi_sections_arr, N_turns, DPhi,
                      R_grid, Z_grid, Phi_grid,
                      BR_flat, BPhi_flat, BZ_flat,
                      wall_R, wall_Z,
                      nx, ny, nz):
    """
    Core njit parallel tracer for multiple Poincaré sections.
    poi_counts shape: (N_seeds * N_sections,) — caller reshapes.
    Output flat layout: [seed0_sec0 pts, seed0_sec1 pts, ..., seed1_sec0 pts, ...]
    """
    TWO_PI = 6.283185307179586
    N_seeds = len(R_seeds)
    N_sections = len(phi_sections_arr)
    n_wall = len(wall_R)
    steps_per_turn = int(round(TWO_PI / DPhi))
    max_pts = N_seeds * N_sections * N_turns

    poi_counts = np.zeros(N_seeds * N_sections, dtype=np.int64)
    poi_R_out  = np.empty(max_pts, dtype=np.float64)
    poi_Z_out  = np.empty(max_pts, dtype=np.float64)

    for s in numba.prange(N_seeds):
        R_pt = R_seeds[s]
        Z_pt = Z_seeds[s]

        if not _point_in_polygon(R_pt, Z_pt, wall_R, wall_Z, n_wall):
            continue

        # Per-section state: turn counts and last-crossing phi
        sec_counts = np.zeros(N_sections, dtype=np.int64)

        phi_pt = 0.0
        total_steps = steps_per_turn * N_turns
        alive = True

        for step in range(total_steps):
            if not alive:
                break
            phi_prev = phi_pt
            R_new, Z_new = _rk4_step(R_pt, Z_pt, phi_pt, DPhi,
                                      R_grid, Z_grid, Phi_grid,
                                      BR_flat, BPhi_flat, BZ_flat,
                                      nx, ny, nz)
            phi_new = phi_prev + DPhi

            if not _point_in_polygon(R_new, Z_new, wall_R, wall_Z, n_wall):
                alive = False
                R_pt = R_new
                Z_pt = Z_new
                phi_pt = phi_new
                break

            # Check each section crossing
            for sec in range(N_sections):
                phi_sec = phi_sections_arr[sec]
                # Detect crossing of phi_sec + turn*2pi
                # Current cumulative phi_prev..phi_new: check if phi_sec (mod 2pi)
                # was crossed in this step
                phi_prev_mod = phi_prev % TWO_PI
                phi_new_mod = phi_new % TWO_PI
                # Check if phi_sec falls in (phi_prev_mod, phi_new_mod]
                # accounting for wrap
                phi_s = phi_sec % TWO_PI
                if phi_s < 0.0:
                    phi_s += TWO_PI

                crossed = False
                if phi_prev_mod < phi_new_mod:
                    crossed = phi_prev_mod < phi_s <= phi_new_mod
                else:
                    # wrap around 2pi
                    crossed = phi_s > phi_prev_mod or phi_s <= phi_new_mod

                if crossed and sec_counts[sec] < N_turns:
                    cnt = sec_counts[sec]
                    flat_idx = (s * N_sections + sec) * N_turns + cnt
                    poi_R_out[flat_idx] = R_new
                    poi_Z_out[flat_idx] = Z_new
                    sec_counts[sec] += 1
                    poi_counts[s * N_sections + sec] = sec_counts[sec]

            R_pt = R_new
            Z_pt = Z_new
            phi_pt = phi_new

        for sec in range(N_sections):
            poi_counts[s * N_sections + sec] = sec_counts[sec]

    return poi_counts, poi_R_out, poi_Z_out


# ---------------------------------------------------------------------------
# Public Python API
# ---------------------------------------------------------------------------

def field_arrays_from_interpolators(itp_BR, itp_BPhi, itp_BZ):
    """
    Extract raw numpy arrays from scipy RegularGridInterpolator for numba use.

    Assumes interpolators are defined on (R_grid, Z_grid, Phi_grid) axes
    (in that order, as passed to RegularGridInterpolator).

    Returns
    -------
    R_grid, Z_grid, Phi_grid : 1-D float64 arrays
    BR_flat, BPhi_flat, BZ_flat : C-contiguous float64 arrays (NR*NZ*NPhi)
    nx, ny, nz : ints
    """
    R_grid   = np.ascontiguousarray(itp_BR.grid[0], dtype=np.float64)
    Z_grid   = np.ascontiguousarray(itp_BR.grid[1], dtype=np.float64)
    Phi_grid = np.ascontiguousarray(itp_BR.grid[2], dtype=np.float64)
    nx, ny, nz = len(R_grid), len(Z_grid), len(Phi_grid)

    BR_flat   = np.ascontiguousarray(itp_BR.values.ravel(),   dtype=np.float64)
    BPhi_flat = np.ascontiguousarray(itp_BPhi.values.ravel(), dtype=np.float64)
    BZ_flat   = np.ascontiguousarray(itp_BZ.values.ravel(),   dtype=np.float64)
    return R_grid, Z_grid, Phi_grid, BR_flat, BPhi_flat, BZ_flat, nx, ny, nz


def precompile_tracer(R_grid, Z_grid, Phi_grid, BR_flat, BPhi_flat, BZ_flat):
    """
    Warm up numba JIT by tracing a trivial 1-seed problem.
    Call once before the real computation to avoid JIT latency.
    """
    nx, ny, nz = len(R_grid), len(Z_grid), len(Phi_grid)
    R_s = np.array([(R_grid[0] + R_grid[-1]) * 0.5])
    Z_s = np.array([(Z_grid[0] + Z_grid[-1]) * 0.5])
    wall_R = np.array([R_grid[0] - 0.1, R_grid[-1] + 0.1,
                       R_grid[-1] + 0.1, R_grid[0] - 0.1, R_grid[0] - 0.1])
    wall_Z = np.array([Z_grid[0] - 0.1, Z_grid[0] - 0.1,
                       Z_grid[-1] + 0.1, Z_grid[-1] + 0.1, Z_grid[0] - 0.1])
    _trace_batch_njit(R_s, Z_s, 0.0, 1, 0.1,
                      R_grid, Z_grid, Phi_grid,
                      BR_flat, BPhi_flat, BZ_flat,
                      wall_R, wall_Z, nx, ny, nz)


def trace_poincare_batch(R_seeds, Z_seeds, phi_section, N_turns, DPhi,
                         R_grid, Z_grid, Phi_grid,
                         BR_flat, BPhi_flat, BZ_flat,
                         wall_R, wall_Z):
    """
    Trace a batch of field-line seeds and record Poincaré crossings.

    Parameters
    ----------
    R_seeds, Z_seeds : 1-D float64 arrays, shape (N_seeds,)
    phi_section : float, toroidal angle of the Poincaré section [rad]
    N_turns : int, number of toroidal turns to trace
    DPhi : float, RK4 step size in phi [rad] (default 0.02)
    R_grid, Z_grid, Phi_grid : 1-D float64 arrays (grid axes)
    BR_flat, BPhi_flat, BZ_flat : flat float64 arrays (NR*NZ*NPhi)
    wall_R, wall_Z : 1-D float64 arrays defining the wall polygon

    Returns
    -------
    poi_counts : int array (N_seeds,) — number of crossings per seed
    poi_R_flat, poi_Z_flat : flat arrays, seed s occupies
        poi_R_flat[s*N_turns : s*N_turns + poi_counts[s]]
    """
    R_seeds   = np.ascontiguousarray(R_seeds,   dtype=np.float64)
    Z_seeds   = np.ascontiguousarray(Z_seeds,   dtype=np.float64)
    R_grid    = np.ascontiguousarray(R_grid,    dtype=np.float64)
    Z_grid    = np.ascontiguousarray(Z_grid,    dtype=np.float64)
    Phi_grid  = np.ascontiguousarray(Phi_grid,  dtype=np.float64)
    BR_flat   = np.ascontiguousarray(BR_flat,   dtype=np.float64)
    BPhi_flat = np.ascontiguousarray(BPhi_flat, dtype=np.float64)
    BZ_flat   = np.ascontiguousarray(BZ_flat,   dtype=np.float64)
    wall_R    = np.ascontiguousarray(wall_R,    dtype=np.float64)
    wall_Z    = np.ascontiguousarray(wall_Z,    dtype=np.float64)

    nx, ny, nz = len(R_grid), len(Z_grid), len(Phi_grid)

    counts, pR, pZ = _trace_batch_njit(
        R_seeds, Z_seeds, float(phi_section), int(N_turns), float(DPhi),
        R_grid, Z_grid, Phi_grid,
        BR_flat, BPhi_flat, BZ_flat,
        wall_R, wall_Z, nx, ny, nz
    )
    return counts, pR, pZ


def trace_poincare_multi_batch(R_seeds, Z_seeds, phi_sections_arr, N_turns, DPhi,
                                R_grid, Z_grid, Phi_grid,
                                BR_flat, BPhi_flat, BZ_flat,
                                wall_R, wall_Z):
    """
    Trace seeds and record Poincaré crossings at multiple phi sections.

    Parameters
    ----------
    phi_sections_arr : 1-D float64 array of section angles [rad]

    Returns
    -------
    poi_counts : int array (N_seeds, N_sections)
    poi_R_flat, poi_Z_flat : flat arrays
        Layout: [seed0_sec0 pts..., seed0_sec1 pts..., seed1_sec0 pts..., ...]
        Seed s, section sec occupies:
            offset = (s * N_sections + sec) * N_turns
            slice  = offset : offset + poi_counts[s, sec]
    """
    R_seeds          = np.ascontiguousarray(R_seeds,          dtype=np.float64)
    Z_seeds          = np.ascontiguousarray(Z_seeds,          dtype=np.float64)
    phi_sections_arr = np.ascontiguousarray(phi_sections_arr, dtype=np.float64)
    R_grid           = np.ascontiguousarray(R_grid,           dtype=np.float64)
    Z_grid           = np.ascontiguousarray(Z_grid,           dtype=np.float64)
    Phi_grid         = np.ascontiguousarray(Phi_grid,         dtype=np.float64)
    BR_flat          = np.ascontiguousarray(BR_flat,          dtype=np.float64)
    BPhi_flat        = np.ascontiguousarray(BPhi_flat,        dtype=np.float64)
    BZ_flat          = np.ascontiguousarray(BZ_flat,          dtype=np.float64)
    wall_R           = np.ascontiguousarray(wall_R,           dtype=np.float64)
    wall_Z           = np.ascontiguousarray(wall_Z,           dtype=np.float64)

    nx, ny, nz = len(R_grid), len(Z_grid), len(Phi_grid)
    N_seeds    = len(R_seeds)
    N_sections = len(phi_sections_arr)

    counts_flat, pR, pZ = _trace_multi_njit(
        R_seeds, Z_seeds, phi_sections_arr, int(N_turns), float(DPhi),
        R_grid, Z_grid, Phi_grid,
        BR_flat, BPhi_flat, BZ_flat,
        wall_R, wall_Z, nx, ny, nz
    )
    poi_counts = counts_flat.reshape(N_seeds, N_sections)
    return poi_counts, pR, pZ
