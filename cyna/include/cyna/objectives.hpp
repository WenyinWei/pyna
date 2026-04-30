#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <thread>
#include <vector>

#include "BS_thread_pool.hpp"
#include "cyna/poincare.hpp"

namespace cyna {

inline bool eval_field_jxb(
    double R,
    double Z,
    double phi,
    const double* BR,
    const double* BPhi,
    const double* BZ,
    const double* R_grid,
    int nR,
    const double* Z_grid,
    int nZ,
    const double* Phi_grid,
    int nPhi,
    double fd_eps_R,
    double fd_eps_Z,
    double fd_eps_phi,
    double& Bmag,
    double& B2,
    double& JxB_mag)
{
    constexpr double mu0 = 4.0e-7 * M_PI;
    phi = mod2pi(phi);

    double br = interp3d(BR, R_grid, nR, Z_grid, nZ, Phi_grid, nPhi, R, Z, phi);
    double bp = interp3d(BPhi, R_grid, nR, Z_grid, nZ, Phi_grid, nPhi, R, Z, phi);
    double bz = interp3d(BZ, R_grid, nR, Z_grid, nZ, Phi_grid, nPhi, R, Z, phi);
    if (!std::isfinite(br) || !std::isfinite(bp) || !std::isfinite(bz) || R <= 1e-12) {
        Bmag = std::numeric_limits<double>::quiet_NaN();
        B2 = std::numeric_limits<double>::quiet_NaN();
        JxB_mag = std::numeric_limits<double>::quiet_NaN();
        return false;
    }

    B2 = br * br + bp * bp + bz * bz;
    Bmag = std::sqrt(B2);

    auto deriv = [&](const double* data, double Rp, double Rm, double Zp, double Zm, double phip, double phim,
                     double& dR, double& dZ, double& dPhi) {
        double fpR = interp3d(data, R_grid, nR, Z_grid, nZ, Phi_grid, nPhi, Rp, Z,  phi);
        double fmR = interp3d(data, R_grid, nR, Z_grid, nZ, Phi_grid, nPhi, Rm, Z,  phi);
        double fpZ = interp3d(data, R_grid, nR, Z_grid, nZ, Phi_grid, nPhi, R,  Zp, phi);
        double fmZ = interp3d(data, R_grid, nR, Z_grid, nZ, Phi_grid, nPhi, R,  Zm, phi);
        double fpP = interp3d(data, R_grid, nR, Z_grid, nZ, Phi_grid, nPhi, R,  Z,  phip);
        double fmP = interp3d(data, R_grid, nR, Z_grid, nZ, Phi_grid, nPhi, R,  Z,  phim);
        if (!std::isfinite(fpR) || !std::isfinite(fmR) ||
            !std::isfinite(fpZ) || !std::isfinite(fmZ) ||
            !std::isfinite(fpP) || !std::isfinite(fmP)) {
            dR = std::numeric_limits<double>::quiet_NaN();
            dZ = std::numeric_limits<double>::quiet_NaN();
            dPhi = std::numeric_limits<double>::quiet_NaN();
            return;
        }
        dR = (fpR - fmR) / (Rp - Rm);
        dZ = (fpZ - fmZ) / (Zp - Zm);
        dPhi = (fpP - fmP) / (phip - phim);
    };

    double epsR = std::max(fd_eps_R, 1e-6);
    double epsZ = std::max(fd_eps_Z, 1e-6);
    double epsP = std::max(fd_eps_phi, 1e-6);
    double Rp = std::min(R + epsR, R_grid[nR - 1]);
    double Rm = std::max(R - epsR, R_grid[0]);
    double Zp = std::min(Z + epsZ, Z_grid[nZ - 1]);
    double Zm = std::max(Z - epsZ, Z_grid[0]);
    double phip = phi + epsP;
    double phim = phi - epsP;

    double dBRdR, dBRdZ, dBRdPhi;
    double dBPdR, dBPdZ, dBPdPhi;
    double dBZdR, dBZdZ, dBZdPhi;
    deriv(BR,   Rp, Rm, Zp, Zm, phip, phim, dBRdR, dBRdZ, dBRdPhi);
    deriv(BPhi, Rp, Rm, Zp, Zm, phip, phim, dBPdR, dBPdZ, dBPdPhi);
    deriv(BZ,   Rp, Rm, Zp, Zm, phip, phim, dBZdR, dBZdZ, dBZdPhi);

    if (!std::isfinite(dBRdR) || !std::isfinite(dBRdZ) || !std::isfinite(dBRdPhi) ||
        !std::isfinite(dBPdR) || !std::isfinite(dBPdZ) || !std::isfinite(dBPdPhi) ||
        !std::isfinite(dBZdR) || !std::isfinite(dBZdZ) || !std::isfinite(dBZdPhi)) {
        JxB_mag = std::numeric_limits<double>::quiet_NaN();
        return false;
    }

    double curl_R = dBZdPhi / R - dBPdZ;
    double curl_Z = (bp + R * dBPdR) / R - dBRdPhi / R;
    double curl_Phi = dBRdZ - dBZdR;

    double JR = curl_R / mu0;
    double JZ = curl_Z / mu0;
    double JPhi = curl_Phi / mu0;

    double cx = JPhi * bz - JZ * bp;
    double cy = JR * bp - JPhi * br;
    double cz = JZ * br - JR * bz;
    JxB_mag = std::sqrt(cx * cx + cy * cy + cz * cz);
    return std::isfinite(JxB_mag);
}

inline void trace_surface_metrics_batch_twall(
    const double* R_seeds,
    const double* Z_seeds,
    int N_seeds,
    double R_axis,
    double Z_axis,
    double phi_start,
    int N_turns,
    double DPhi,
    const double* BR,
    const double* BPhi,
    const double* BZ,
    const double* R_grid,
    int nR,
    const double* Z_grid,
    int nZ,
    const double* Phi_grid,
    int nPhi,
    const double* wall_phi_centers,
    int n_phi_wall,
    const double* wall_R,
    const double* wall_Z,
    int n_theta_wall,
    double fd_eps_R,
    double fd_eps_Z,
    double fd_eps_phi,
    int n_threads,
    double* iota_out,
    double* Bmean_out,
    double* B2mean_out,
    double* Bmin_out,
    double* Bmax_out,
    double* JxBmean_out,
    double* turns_out,
    int* alive_out)
{
    if (n_threads <= 0)
        n_threads = (int)std::thread::hardware_concurrency();

    BS::thread_pool pool((unsigned int)n_threads);
    pool.parallelize_loop(0, N_seeds, [&](int i_start, int i_end) {
        for (int i = i_start; i < i_end; ++i) {
            double R = R_seeds[i];
            double Z = Z_seeds[i];
            double phi = phi_start;
            double phi_end = phi_start + N_turns * 2.0 * M_PI;
            double theta_prev = std::atan2(Z - Z_axis, R - R_axis);
            double theta_acc = 0.0;
            double Bsum = 0.0;
            double B2sum = 0.0;
            double JxBsum = 0.0;
            double Bmin = std::numeric_limits<double>::infinity();
            double Bmax = 0.0;
            int n_samples = 0;
            bool alive = true;

            while (phi < phi_end - 1e-12) {
                double Bmag, B2, JxB_mag;
                bool ok = eval_field_jxb(
                    R, Z, phi,
                    BR, BPhi, BZ,
                    R_grid, nR, Z_grid, nZ, Phi_grid, nPhi,
                    fd_eps_R, fd_eps_Z, fd_eps_phi,
                    Bmag, B2, JxB_mag);
                if (!ok) {
                    alive = false;
                    break;
                }

                Bsum += Bmag;
                B2sum += B2;
                JxBsum += JxB_mag;
                Bmin = std::min(Bmin, Bmag);
                Bmax = std::max(Bmax, Bmag);
                ++n_samples;

                double step = std::min(DPhi, phi_end - phi);
                double R_next = R;
                double Z_next = Z;
                rk4_step(R_next, Z_next, phi, step,
                         BR, BPhi, BZ,
                         R_grid, nR, Z_grid, nZ, Phi_grid, nPhi);
                double phi_next = phi + step;
                if (!std::isfinite(R_next) || !std::isfinite(Z_next) ||
                    R_next < R_grid[0] || R_next > R_grid[nR - 1] ||
                    Z_next < Z_grid[0] || Z_next > Z_grid[nZ - 1] ||
                    !point_in_toroidal_wall(R_next, Z_next, phi_next,
                                            wall_phi_centers, wall_R, wall_Z,
                                            n_phi_wall, n_theta_wall)) {
                    alive = false;
                    break;
                }

                double theta_next = std::atan2(Z_next - Z_axis, R_next - R_axis);
                double dtheta = theta_next - theta_prev;
                while (dtheta > M_PI) dtheta -= 2.0 * M_PI;
                while (dtheta < -M_PI) dtheta += 2.0 * M_PI;
                theta_acc += dtheta;
                theta_prev = theta_next;

                R = R_next;
                Z = Z_next;
                phi = phi_next;
            }

            double turns = std::max(0.0, (phi - phi_start) / (2.0 * M_PI));
            turns_out[i] = turns;
            alive_out[i] = alive ? 1 : 0;
            if (n_samples <= 0 || turns <= 1e-12) {
                iota_out[i] = std::numeric_limits<double>::quiet_NaN();
                Bmean_out[i] = std::numeric_limits<double>::quiet_NaN();
                B2mean_out[i] = std::numeric_limits<double>::quiet_NaN();
                Bmin_out[i] = std::numeric_limits<double>::quiet_NaN();
                Bmax_out[i] = std::numeric_limits<double>::quiet_NaN();
                JxBmean_out[i] = std::numeric_limits<double>::quiet_NaN();
                continue;
            }

            iota_out[i] = std::abs(theta_acc / (2.0 * M_PI * turns));
            Bmean_out[i] = Bsum / n_samples;
            B2mean_out[i] = B2sum / n_samples;
            Bmin_out[i] = Bmin;
            Bmax_out[i] = Bmax;
            JxBmean_out[i] = JxBsum / n_samples;
        }
    }).wait();
}

inline void summarize_profile_objectives(
    const double* r_eff,
    const double* iota,
    const double* B_mean,
    const double* B2_mean,
    const double* B_min,
    const double* B_max,
    const double* JxB_mean,
    int N,
    double a_eff,
    double* mean_iota_prime_out,
    double* mean_abs_iota_prime_out,
    double* eps_eff_out,
    double* Bvol_avg_out,
    double* beta_max_fast_out,
    double* force_balance_out,
    double* magnetic_pressure_out,
    int* n_valid_out,
    double* D_Merc_proxy_out = nullptr)
{
    constexpr double mu0 = 4.0e-7 * M_PI;
    struct Entry {
        double r;
        double i;
        double bm;
        double b2;
        double bmin;
        double bmax;
        double jxb;
    };

    std::vector<Entry> v;
    v.reserve(N);
    for (int k = 0; k < N; ++k) {
        if (!std::isfinite(r_eff[k]) || !std::isfinite(iota[k]) ||
            !std::isfinite(B_mean[k]) || !std::isfinite(B2_mean[k]) ||
            !std::isfinite(B_min[k]) || !std::isfinite(B_max[k]) ||
            !std::isfinite(JxB_mean[k]) || r_eff[k] <= 0.0) {
            continue;
        }
        v.push_back({r_eff[k], iota[k], B_mean[k], B2_mean[k], B_min[k], B_max[k], JxB_mean[k]});
    }

    std::sort(v.begin(), v.end(), [](const Entry& a, const Entry& b) { return a.r < b.r; });
    int M = (int)v.size();
    *n_valid_out = M;
    if (M == 0 || a_eff <= 0.0) {
        *mean_iota_prime_out = std::numeric_limits<double>::quiet_NaN();
        *mean_abs_iota_prime_out = std::numeric_limits<double>::quiet_NaN();
        *eps_eff_out = std::numeric_limits<double>::quiet_NaN();
        *Bvol_avg_out = std::numeric_limits<double>::quiet_NaN();
        *beta_max_fast_out = std::numeric_limits<double>::quiet_NaN();
        *force_balance_out = std::numeric_limits<double>::quiet_NaN();
        *magnetic_pressure_out = std::numeric_limits<double>::quiet_NaN();
        if (D_Merc_proxy_out) *D_Merc_proxy_out = std::numeric_limits<double>::quiet_NaN();
        return;
    }

    std::vector<double> w(M, 0.0);
    for (int i = 0; i < M; ++i) {
        double lo = (i == 0) ? 0.0 : 0.5 * (v[i - 1].r + v[i].r);
        double hi = (i == M - 1) ? a_eff : 0.5 * (v[i].r + v[i + 1].r);
        lo = std::max(0.0, std::min(lo, a_eff));
        hi = std::max(lo, std::min(hi, a_eff));
        w[i] = std::max(0.0, hi * hi - lo * lo);
    }
    double wsum = 0.0;
    for (double wi : w) wsum += wi;
    if (wsum <= 0.0) {
        for (double& wi : w) wi = 1.0 / std::max(1, M);
    } else {
        for (double& wi : w) wi /= wsum;
    }

    std::vector<double> shear(M, 0.0);
    if (M >= 2) {
        for (int i = 0; i < M; ++i) {
            if (i == 0) {
                shear[i] = (v[1].i - v[0].i) / std::max(v[1].r - v[0].r, 1e-12);
            } else if (i == M - 1) {
                shear[i] = (v[M - 1].i - v[M - 2].i) / std::max(v[M - 1].r - v[M - 2].r, 1e-12);
            } else {
                shear[i] = (v[i + 1].i - v[i - 1].i) / std::max(v[i + 1].r - v[i - 1].r, 1e-12);
            }
        }
    } else {
        shear[0] = 0.0;
    }

    double mean_shear = 0.0;
    double mean_abs_shear = 0.0;
    double eps_eff = 0.0;
    double Bvol = 0.0;
    double B2vol = 0.0;
    double force_balance = 0.0;
    for (int i = 0; i < M; ++i) {
        mean_shear += w[i] * shear[i];
        mean_abs_shear += w[i] * std::abs(shear[i]);
        double eps_h = (v[i].bmax - v[i].bmin) / std::max(v[i].bmax + v[i].bmin, 1e-30);
        eps_h = std::max(0.0, eps_h);
        eps_eff += w[i] * (0.64 * std::pow(eps_h, 1.5));
        Bvol += w[i] * v[i].bm;
        B2vol += w[i] * v[i].b2;
        force_balance += w[i] * v[i].jxb;
    }

    *mean_iota_prime_out = mean_shear;
    *mean_abs_iota_prime_out = mean_abs_shear;
    *eps_eff_out = eps_eff;
    *Bvol_avg_out = Bvol;
    *force_balance_out = force_balance;
    *magnetic_pressure_out = 0.5 * B2vol / mu0;
    double beta_fast = (B2vol > 0.0) ? (2.0 * mu0 * a_eff * force_balance / B2vol) : std::numeric_limits<double>::quiet_NaN();
    if (std::isfinite(beta_fast))
        beta_fast = std::max(0.0, beta_fast);
    *beta_max_fast_out = beta_fast;

    // Mercier stability proxy: D_Merc_proxy = weighted mean of (s_norm² - 0.25)
    // where s_norm = r * d(iota)/dr / iota  (normalized magnetic shear)
    // D_Merc_proxy < 0 indicates Mercier stability
    if (D_Merc_proxy_out) {
        double d_merc = 0.0;
        for (int i = 0; i < M; ++i) {
            double iota_val = std::abs(v[i].i);
            double s_norm = (iota_val > 1e-12) ? (v[i].r * shear[i] / iota_val) : 0.0;
            d_merc += w[i] * (s_norm * s_norm - 0.25);
        }
        *D_Merc_proxy_out = d_merc;
    }
}

}  // namespace cyna
