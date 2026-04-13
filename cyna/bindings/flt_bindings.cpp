// flt_bindings.cpp — pybind11 bindings for cyna Poincaré tracer + coil fields
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <thread>
#include <stdexcept>
#include <cmath>
#include <vector>
#include "cyna/poincare.hpp"
#include "cyna/objectives.hpp"
#include "cyna/coil_field.hpp"

#ifdef CYNA_CUDA_ENABLED
extern "C" bool cyna_coil_circular_field_cuda(
    float,float,float, float,float,float, float,float,
    const float*,int, float*,float*,float*);
extern "C" bool cyna_coil_biot_savart_cuda(
    const float*,const float*,int,float,
    const float*,int, float*,float*,float*);
#endif

namespace py = pybind11;

// Helper: assert contiguous double array
static const double* buf(const py::array_t<double>& a, const char* name) {
    if (!(a.flags() & py::array::c_style))
        throw std::runtime_error(std::string(name) + " must be C-contiguous");
    return a.data();
}

// ---------------------------------------------------------------------------
// compute_A_matrix_batch
// ---------------------------------------------------------------------------
static py::array_t<double> py_compute_A_matrix_batch(
    py::array_t<double> R_arr,
    py::array_t<double> Z_arr,
    py::array_t<double> phi_arr,
    py::array_t<double> BR,
    py::array_t<double> BPhi,
    py::array_t<double> BZ,
    py::array_t<double> R_grid,
    py::array_t<double> Z_grid,
    py::array_t<double> Phi_grid,
    double eps)
{
    int N    = (int)R_arr.size();
    int nR   = (int)R_grid.size();
    int nZ   = (int)Z_grid.size();
    int nPhi = (int)Phi_grid.size();

    const double* pR    = R_arr.data();
    const double* pZ    = Z_arr.data();
    const double* pPhi  = phi_arr.data();
    const double* pBR   = buf(BR,       "BR");
    const double* pBPhi = buf(BPhi,     "BPhi");
    const double* pBZ   = buf(BZ,       "BZ");
    const double* pRg   = buf(R_grid,   "R_grid");
    const double* pZg   = buf(Z_grid,   "Z_grid");
    const double* pPg   = buf(Phi_grid, "Phi_grid");

    // Output: shape (N, 2, 2), C-order
    py::array_t<double> out({N, 2, 2});
    double* pOut = out.mutable_data();

    // Lambda: evaluate g = [R*BR/BPhi, R*BZ/BPhi] at (R, Z, phi)
    auto g_eval = [&](double R, double Z, double phi, double& g0, double& g1) {
        double bp = cyna::interp3d(pBPhi, pRg, nR, pZg, nZ, pPg, nPhi, R, Z, phi);
        if (!std::isfinite(bp) || std::abs(bp) < 1e-30) {
            g0 = std::numeric_limits<double>::quiet_NaN();
            g1 = std::numeric_limits<double>::quiet_NaN();
            return;
        }
        double br = cyna::interp3d(pBR, pRg, nR, pZg, nZ, pPg, nPhi, R, Z, phi);
        double bz = cyna::interp3d(pBZ, pRg, nR, pZg, nZ, pPg, nPhi, R, Z, phi);
        g0 = R * br / bp;
        g1 = R * bz / bp;
    };

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int k = 0; k < N; ++k) {
        double R   = pR[k];
        double Z   = pZ[k];
        double phi = pPhi[k];

        double g0Rp, g1Rp, g0Rm, g1Rm;
        double g0Zp, g1Zp, g0Zm, g1Zm;
        g_eval(R + eps, Z,       phi, g0Rp, g1Rp);
        g_eval(R - eps, Z,       phi, g0Rm, g1Rm);
        g_eval(R,       Z + eps, phi, g0Zp, g1Zp);
        g_eval(R,       Z - eps, phi, g0Zm, g1Zm);

        double inv2eps = 1.0 / (2.0 * eps);
        // Row-major: [k,0,0], [k,0,1], [k,1,0], [k,1,1]
        pOut[k * 4 + 0] = (g0Rp - g0Rm) * inv2eps;  // dg0/dR
        pOut[k * 4 + 1] = (g0Zp - g0Zm) * inv2eps;  // dg0/dZ
        pOut[k * 4 + 2] = (g1Rp - g1Rm) * inv2eps;  // dg1/dR
        pOut[k * 4 + 3] = (g1Zp - g1Zm) * inv2eps;  // dg1/dZ
    }

    return out;
}

static py::tuple py_trace_poincare_batch(
    py::array_t<double> R_seeds,
    py::array_t<double> Z_seeds,
    double phi_section,
    int N_turns,
    double DPhi,
    py::array_t<double> BR,
    py::array_t<double> BPhi,
    py::array_t<double> BZ,
    py::array_t<double> R_grid,
    py::array_t<double> Z_grid,
    py::array_t<double> Phi_grid,
    py::array_t<double> wall_R,
    py::array_t<double> wall_Z,
    int n_threads)
{
    if (n_threads <= 0)
        n_threads = (int)std::thread::hardware_concurrency();

    int N_seeds = (int)R_seeds.size();
    int nR      = (int)R_grid.size();
    int nZ      = (int)Z_grid.size();
    int nPhi    = (int)Phi_grid.size();
    int n_wall  = (int)wall_R.size();

    // Allocate outputs
    py::array_t<int>    poi_counts({ N_seeds });
    py::array_t<double> poi_R_flat({ N_seeds * N_turns });
    py::array_t<double> poi_Z_flat({ N_seeds * N_turns });

    std::fill(poi_counts.mutable_data(), poi_counts.mutable_data() + N_seeds, 0);
    std::fill(poi_R_flat.mutable_data(), poi_R_flat.mutable_data() + N_seeds * N_turns, 0.0);
    std::fill(poi_Z_flat.mutable_data(), poi_Z_flat.mutable_data() + N_seeds * N_turns, 0.0);

    cyna::trace_poincare_batch(
        buf(R_seeds, "R_seeds"), buf(Z_seeds, "Z_seeds"), N_seeds,
        phi_section, N_turns, DPhi,
        buf(BR, "BR"), buf(BPhi, "BPhi"), buf(BZ, "BZ"),
        buf(R_grid, "R_grid"), nR,
        buf(Z_grid, "Z_grid"), nZ,
        buf(Phi_grid, "Phi_grid"), nPhi,
        buf(wall_R, "wall_R"), buf(wall_Z, "wall_Z"), n_wall,
        n_threads,
        poi_counts.mutable_data(),
        poi_R_flat.mutable_data(),
        poi_Z_flat.mutable_data());

    return py::make_tuple(poi_counts, poi_R_flat, poi_Z_flat);
}

static py::tuple py_trace_poincare_batch_twall(
    py::array_t<double> R_seeds,
    py::array_t<double> Z_seeds,
    double phi_section,
    int N_turns,
    double DPhi,
    py::array_t<double> BR,
    py::array_t<double> BPhi,
    py::array_t<double> BZ,
    py::array_t<double> R_grid,
    py::array_t<double> Z_grid,
    py::array_t<double> Phi_grid,
    py::array_t<double> wall_phi_centers,
    py::array_t<double> wall_R,
    py::array_t<double> wall_Z,
    int n_threads)
{
    if (n_threads <= 0)
        n_threads = (int)std::thread::hardware_concurrency();

    if (wall_R.ndim() != 2 || wall_Z.ndim() != 2)
        throw std::runtime_error("wall_R and wall_Z must be 2-D arrays [n_phi_wall, n_theta_wall]");
    if (wall_R.shape(0) != wall_Z.shape(0) || wall_R.shape(1) != wall_Z.shape(1))
        throw std::runtime_error("wall_R and wall_Z shapes must match");
    if ((int)wall_phi_centers.size() != (int)wall_R.shape(0))
        throw std::runtime_error("wall_phi_centers length must equal wall_R.shape[0]");

    int N_seeds = (int)R_seeds.size();
    int nR      = (int)R_grid.size();
    int nZ      = (int)Z_grid.size();
    int nPhi    = (int)Phi_grid.size();
    int n_phi_wall   = (int)wall_R.shape(0);
    int n_theta_wall = (int)wall_R.shape(1);

    py::array_t<int>    poi_counts({ N_seeds });
    py::array_t<double> poi_R_flat({ N_seeds * N_turns });
    py::array_t<double> poi_Z_flat({ N_seeds * N_turns });

    std::fill(poi_counts.mutable_data(), poi_counts.mutable_data() + N_seeds, 0);
    std::fill(poi_R_flat.mutable_data(), poi_R_flat.mutable_data() + N_seeds * N_turns, 0.0);
    std::fill(poi_Z_flat.mutable_data(), poi_Z_flat.mutable_data() + N_seeds * N_turns, 0.0);

    cyna::trace_poincare_batch_twall(
        buf(R_seeds, "R_seeds"), buf(Z_seeds, "Z_seeds"), N_seeds,
        phi_section, N_turns, DPhi,
        buf(BR, "BR"), buf(BPhi, "BPhi"), buf(BZ, "BZ"),
        buf(R_grid, "R_grid"), nR,
        buf(Z_grid, "Z_grid"), nZ,
        buf(Phi_grid, "Phi_grid"), nPhi,
        buf(wall_phi_centers, "wall_phi_centers"), n_phi_wall,
        buf(wall_R, "wall_R"), buf(wall_Z, "wall_Z"), n_theta_wall,
        n_threads,
        poi_counts.mutable_data(),
        poi_R_flat.mutable_data(),
        poi_Z_flat.mutable_data());

    return py::make_tuple(poi_counts, poi_R_flat, poi_Z_flat);
}

static py::tuple py_trace_poincare_multi(
    py::array_t<double> R_seeds,
    py::array_t<double> Z_seeds,
    py::array_t<double> phi_sections,
    int N_turns,
    double DPhi,
    py::array_t<double> BR,
    py::array_t<double> BPhi,
    py::array_t<double> BZ,
    py::array_t<double> R_grid,
    py::array_t<double> Z_grid,
    py::array_t<double> Phi_grid,
    py::array_t<double> wall_R,
    py::array_t<double> wall_Z,
    int n_threads)
{
    if (n_threads <= 0)
        n_threads = (int)std::thread::hardware_concurrency();

    int N_seeds = (int)R_seeds.size();
    int n_sec   = (int)phi_sections.size();
    int nR      = (int)R_grid.size();
    int nZ      = (int)Z_grid.size();
    int nPhi    = (int)Phi_grid.size();
    int n_wall  = (int)wall_R.size();

    // Outputs
    py::array_t<int>    poi_counts({ N_seeds * n_sec });
    py::array_t<double> poi_R_flat({ N_seeds * N_turns * n_sec });
    py::array_t<double> poi_Z_flat({ N_seeds * N_turns * n_sec });

    std::fill(poi_counts.mutable_data(), poi_counts.mutable_data() + N_seeds * n_sec, 0);
    std::fill(poi_R_flat.mutable_data(), poi_R_flat.mutable_data() + N_seeds * N_turns * n_sec, 0.0);
    std::fill(poi_Z_flat.mutable_data(), poi_Z_flat.mutable_data() + N_seeds * N_turns * n_sec, 0.0);

    // Normalise sections to [0, 2pi)
    std::vector<double> secs(n_sec);
    for (int s = 0; s < n_sec; ++s)
        secs[s] = cyna::mod2pi(phi_sections.data()[s]);

    int nt = n_threads;
    BS::thread_pool pool((unsigned int)nt);
    pool.parallelize_loop(0, N_seeds, [&](int i_start, int i_end) {
        for (int i = i_start; i < i_end; ++i) {
            cyna::trace_one_seed(
                i, N_seeds,
                R_seeds.data()[i], Z_seeds.data()[i], secs[0],
                secs.data(), n_sec,
                N_turns, DPhi,
                buf(BR, "BR"), buf(BPhi, "BPhi"), buf(BZ, "BZ"),
                buf(R_grid, "R_grid"), nR,
                buf(Z_grid, "Z_grid"), nZ,
                buf(Phi_grid, "Phi_grid"), nPhi,
                buf(wall_R, "wall_R"), buf(wall_Z, "wall_Z"), n_wall,
                poi_counts.mutable_data(),
                poi_R_flat.mutable_data(),
                poi_Z_flat.mutable_data());
        }
    }).wait();

    // Reshape poi_counts to [N_seeds, n_sec]
    poi_counts.resize({ N_seeds, n_sec });
    return py::make_tuple(poi_counts, poi_R_flat, poi_Z_flat);
}

static py::tuple py_trace_surface_metrics_batch_twall(
    py::array_t<double> R_seeds,
    py::array_t<double> Z_seeds,
    double R_axis,
    double Z_axis,
    double phi_start,
    int N_turns,
    double DPhi,
    py::array_t<double> BR,
    py::array_t<double> BPhi,
    py::array_t<double> BZ,
    py::array_t<double> R_grid,
    py::array_t<double> Z_grid,
    py::array_t<double> Phi_grid,
    py::array_t<double> wall_phi_centers,
    py::array_t<double> wall_R,
    py::array_t<double> wall_Z,
    double fd_eps_R,
    double fd_eps_Z,
    double fd_eps_phi,
    int n_threads)
{
    if (n_threads <= 0)
        n_threads = (int)std::thread::hardware_concurrency();
    if (wall_R.ndim() != 2 || wall_Z.ndim() != 2)
        throw std::runtime_error("wall_R and wall_Z must be 2-D arrays [n_phi_wall, n_theta_wall]");
    if (wall_R.shape(0) != wall_Z.shape(0) || wall_R.shape(1) != wall_Z.shape(1))
        throw std::runtime_error("wall_R and wall_Z shapes must match");
    if ((int)wall_phi_centers.size() != (int)wall_R.shape(0))
        throw std::runtime_error("wall_phi_centers length must equal wall_R.shape[0]");

    int N_seeds = (int)R_seeds.size();
    int n_phi_wall = (int)wall_R.shape(0);
    int n_theta_wall = (int)wall_R.shape(1);

    py::array_t<double> iota({N_seeds});
    py::array_t<double> B_mean({N_seeds});
    py::array_t<double> B2_mean({N_seeds});
    py::array_t<double> B_min({N_seeds});
    py::array_t<double> B_max({N_seeds});
    py::array_t<double> JxB_mean({N_seeds});
    py::array_t<double> turns({N_seeds});
    py::array_t<int> alive({N_seeds});

    cyna::trace_surface_metrics_batch_twall(
        buf(R_seeds, "R_seeds"), buf(Z_seeds, "Z_seeds"), N_seeds,
        R_axis, Z_axis, phi_start, N_turns, DPhi,
        buf(BR, "BR"), buf(BPhi, "BPhi"), buf(BZ, "BZ"),
        buf(R_grid, "R_grid"), (int)R_grid.size(),
        buf(Z_grid, "Z_grid"), (int)Z_grid.size(),
        buf(Phi_grid, "Phi_grid"), (int)Phi_grid.size(),
        buf(wall_phi_centers, "wall_phi_centers"), n_phi_wall,
        buf(wall_R, "wall_R"), buf(wall_Z, "wall_Z"), n_theta_wall,
        fd_eps_R, fd_eps_Z, fd_eps_phi, n_threads,
        iota.mutable_data(), B_mean.mutable_data(), B2_mean.mutable_data(),
        B_min.mutable_data(), B_max.mutable_data(), JxB_mean.mutable_data(),
        turns.mutable_data(), alive.mutable_data());

    return py::make_tuple(iota, B_mean, B2_mean, B_min, B_max, JxB_mean, turns, alive);
}

static py::tuple py_summarize_profile_objectives(
    py::array_t<double> r_eff,
    py::array_t<double> iota,
    py::array_t<double> B_mean,
    py::array_t<double> B2_mean,
    py::array_t<double> B_min,
    py::array_t<double> B_max,
    py::array_t<double> JxB_mean,
    double a_eff)
{
    py::ssize_t N = r_eff.size();
    if (iota.size() != N || B_mean.size() != N || B2_mean.size() != N ||
        B_min.size() != N || B_max.size() != N || JxB_mean.size() != N)
        throw std::runtime_error("profile arrays must have identical length");

    double mean_iota_prime = std::numeric_limits<double>::quiet_NaN();
    double mean_abs_iota_prime = std::numeric_limits<double>::quiet_NaN();
    double eps_eff = std::numeric_limits<double>::quiet_NaN();
    double Bvol_avg = std::numeric_limits<double>::quiet_NaN();
    double beta_max_fast = std::numeric_limits<double>::quiet_NaN();
    double force_balance = std::numeric_limits<double>::quiet_NaN();
    double magnetic_pressure = std::numeric_limits<double>::quiet_NaN();
    double D_Merc_proxy = std::numeric_limits<double>::quiet_NaN();
    int n_valid = 0;

    cyna::summarize_profile_objectives(
        buf(r_eff, "r_eff"), buf(iota, "iota"),
        buf(B_mean, "B_mean"), buf(B2_mean, "B2_mean"),
        buf(B_min, "B_min"), buf(B_max, "B_max"),
        buf(JxB_mean, "JxB_mean"), (int)N, a_eff,
        &mean_iota_prime, &mean_abs_iota_prime, &eps_eff, &Bvol_avg,
        &beta_max_fast, &force_balance, &magnetic_pressure, &n_valid,
        &D_Merc_proxy);

    return py::make_tuple(mean_iota_prime, mean_abs_iota_prime, eps_eff, Bvol_avg,
                          beta_max_fast, force_balance, magnetic_pressure, n_valid,
                          D_Merc_proxy);
}

static py::tuple py_trace_poincare_beta_sweep(
    py::array_t<double> R_seeds,
    py::array_t<double> Z_seeds,
    py::array_t<double> phi_sections,
    int N_turns,
    double DPhi,
    py::array_t<double> BR,
    py::array_t<double> BPhi,
    py::array_t<double> BZ,
    py::array_t<double> R_grid,
    py::array_t<double> Z_grid,
    py::array_t<double> Phi_grid,
    py::array_t<double> wall_R,
    py::array_t<double> wall_Z,
    double beta,
    double R_ax,
    double Z_ax,
    double a_eff,
    double alpha_pressure,
    double B_ref,
    int n_threads)
{
    if (n_threads <= 0)
        n_threads = (int)std::thread::hardware_concurrency();

    int N_seeds = (int)R_seeds.size();
    int n_sec   = (int)phi_sections.size();
    int nR      = (int)R_grid.size();
    int nZ      = (int)Z_grid.size();
    int nPhi    = (int)Phi_grid.size();
    int n_wall  = (int)wall_R.size();

    py::array_t<int>    poi_counts({ N_seeds, n_sec });
    py::array_t<double> poi_R({ N_seeds, n_sec, N_turns });
    py::array_t<double> poi_Z({ N_seeds, n_sec, N_turns });

    std::fill(poi_counts.mutable_data(), poi_counts.mutable_data() + N_seeds * n_sec, 0);
    const double NAN_VAL = std::numeric_limits<double>::quiet_NaN();
    std::fill(poi_R.mutable_data(), poi_R.mutable_data() + N_seeds * n_sec * N_turns, NAN_VAL);
    std::fill(poi_Z.mutable_data(), poi_Z.mutable_data() + N_seeds * n_sec * N_turns, NAN_VAL);

    cyna::trace_poincare_beta_sweep(
        buf(R_seeds, "R_seeds"), buf(Z_seeds, "Z_seeds"), N_seeds,
        buf(phi_sections, "phi_sections"), n_sec,
        N_turns, DPhi,
        buf(BR, "BR"), buf(BPhi, "BPhi"), buf(BZ, "BZ"),
        buf(R_grid, "R_grid"), nR,
        buf(Z_grid, "Z_grid"), nZ,
        buf(Phi_grid, "Phi_grid"), nPhi,
        buf(wall_R, "wall_R"), buf(wall_Z, "wall_Z"), n_wall,
        beta, R_ax, Z_ax, a_eff, alpha_pressure, B_ref,
        n_threads,
        poi_counts.mutable_data(),
        poi_R.mutable_data(),
        poi_Z.mutable_data());

    return py::make_tuple(poi_R, poi_Z, poi_counts);
}

PYBIND11_MODULE(_cyna_ext, m) {
    m.attr("__version__") = "0.2.0";
    m.attr("available")   = true;

    // Debug: expose point_in_wall directly
    m.def("point_in_wall", [](double R, double Z,
                               py::array_t<double> wR, py::array_t<double> wZ) {
        return cyna::point_in_wall(R, Z, wR.data(), wZ.data(), (int)wR.size());
    });

    // Debug: one RK4 step
    m.def("rk4_step_test", [](double R, double Z, double phi, double DPhi,
                               py::array_t<double> BR, py::array_t<double> BPhi,
                               py::array_t<double> BZ,
                               py::array_t<double> Rg, py::array_t<double> Zg,
                               py::array_t<double> Pg) {
        cyna::rk4_step(R, Z, phi, DPhi,
            BR.data(), BPhi.data(), BZ.data(),
            Rg.data(), (int)Rg.size(),
            Zg.data(), (int)Zg.size(),
            Pg.data(), (int)Pg.size());
        return py::make_tuple(R, Z);
    });

    m.def("interp3d_test", [](double R, double Z, double Phi,
                               py::array_t<double> data,
                               py::array_t<double> Rg, py::array_t<double> Zg,
                               py::array_t<double> Pg) {
        return cyna::interp3d(data.data(),
            Rg.data(), (int)Rg.size(),
            Zg.data(), (int)Zg.size(),
            Pg.data(), (int)Pg.size(),
            R, Z, Phi);
    });

    m.def("trace_poincare_batch", &py_trace_poincare_batch,
        py::arg("R_seeds"), py::arg("Z_seeds"), py::arg("phi_section"),
        py::arg("N_turns"), py::arg("DPhi"),
        py::arg("BR"), py::arg("BPhi"), py::arg("BZ"),
        py::arg("R_grid"), py::arg("Z_grid"), py::arg("Phi_grid"),
        py::arg("wall_R"), py::arg("wall_Z"),
        py::arg("n_threads") = -1,
        "Trace Poincaré section for multiple seeds against a fixed 2-D wall slice.\n"
        "Returns (poi_counts, poi_R_flat, poi_Z_flat).");

    m.def("trace_poincare_batch_twall", &py_trace_poincare_batch_twall,
        py::arg("R_seeds"), py::arg("Z_seeds"), py::arg("phi_section"),
        py::arg("N_turns"), py::arg("DPhi"),
        py::arg("BR"), py::arg("BPhi"), py::arg("BZ"),
        py::arg("R_grid"), py::arg("Z_grid"), py::arg("Phi_grid"),
        py::arg("wall_phi_centers"), py::arg("wall_R"), py::arg("wall_Z"),
        py::arg("n_threads") = -1,
        "Trace Poincaré section for multiple seeds against a toroidally varying wall.\n"
        "Returns (poi_counts, poi_R_flat, poi_Z_flat).");

    m.def("trace_poincare_multi", &py_trace_poincare_multi,
        py::arg("R_seeds"), py::arg("Z_seeds"), py::arg("phi_sections"),
        py::arg("N_turns"), py::arg("DPhi"),
        py::arg("BR"), py::arg("BPhi"), py::arg("BZ"),
        py::arg("R_grid"), py::arg("Z_grid"), py::arg("Phi_grid"),
        py::arg("wall_R"), py::arg("wall_Z"),
        py::arg("n_threads") = -1,
        "Trace Poincaré sections for multiple seeds and multiple phi sections.\n"
        "Returns (poi_counts [N_seeds x n_sec], poi_R_flat, poi_Z_flat).");

    m.def("trace_connection_length_twall",
        [](py::array_t<double> R_seeds, py::array_t<double> Z_seeds,
           double phi_start, int max_turns, double DPhi,
           py::array_t<double> BR, py::array_t<double> BPhi, py::array_t<double> BZ,
           py::array_t<double> R_grid, py::array_t<double> Z_grid,
           py::array_t<double> Phi_grid,
           py::array_t<double> wall_phi_centers,
           py::array_t<double> wall_R, py::array_t<double> wall_Z,
           int n_threads) -> py::tuple
        {
            if (n_threads <= 0) n_threads = (int)std::thread::hardware_concurrency();
            if (wall_R.ndim() != 2 || wall_Z.ndim() != 2)
                throw std::runtime_error("wall_R and wall_Z must be 2-D");
            if (wall_R.shape(0) != wall_Z.shape(0) || wall_R.shape(1) != wall_Z.shape(1))
                throw std::runtime_error("wall_R and wall_Z shapes must match");
            if ((int)wall_phi_centers.size() != (int)wall_R.shape(0))
                throw std::runtime_error("wall_phi_centers length must equal wall_R.shape[0]");

            int N_seeds      = (int)R_seeds.size();
            int n_phi_wall   = (int)wall_R.shape(0);
            int n_theta_wall = (int)wall_R.shape(1);

            py::array_t<double> L_fwd({ N_seeds });
            py::array_t<double> L_bwd({ N_seeds });

            cyna::trace_connection_length_twall(
                buf(R_seeds,"R_seeds"), buf(Z_seeds,"Z_seeds"), N_seeds,
                phi_start, max_turns, DPhi,
                buf(BR,"BR"), buf(BPhi,"BPhi"), buf(BZ,"BZ"),
                buf(R_grid,"R_grid"), (int)R_grid.size(),
                buf(Z_grid,"Z_grid"), (int)Z_grid.size(),
                buf(Phi_grid,"Phi_grid"), (int)Phi_grid.size(),
                buf(wall_phi_centers,"wall_phi_centers"), n_phi_wall,
                buf(wall_R,"wall_R"), buf(wall_Z,"wall_Z"), n_theta_wall,
                n_threads,
                L_fwd.mutable_data(),
                L_bwd.mutable_data());

            return py::make_tuple(L_fwd, L_bwd);
        },
        py::arg("R_seeds"), py::arg("Z_seeds"), py::arg("phi_start"),
        py::arg("max_turns"), py::arg("DPhi"),
        py::arg("BR"), py::arg("BPhi"), py::arg("BZ"),
        py::arg("R_grid"), py::arg("Z_grid"), py::arg("Phi_grid"),
        py::arg("wall_phi_centers"), py::arg("wall_R"), py::arg("wall_Z"),
        py::arg("n_threads") = -1,
        "Compute connection length (forward + backward) for each seed against toroidal wall.\n"
        "Returns (L_fwd, L_bwd) in metres; sentinel=1e30 means no termination within max_turns.");

    m.def("trace_wall_hits_twall",
        [](py::array_t<double> R_seeds, py::array_t<double> Z_seeds,
           double phi_start, int max_turns, double DPhi,
           py::array_t<double> BR, py::array_t<double> BPhi, py::array_t<double> BZ,
           py::array_t<double> R_grid, py::array_t<double> Z_grid,
           py::array_t<double> Phi_grid,
           py::array_t<double> wall_phi_centers,
           py::array_t<double> wall_R, py::array_t<double> wall_Z,
           int n_threads) -> py::tuple
        {
            if (n_threads <= 0) n_threads = (int)std::thread::hardware_concurrency();
            if (wall_R.ndim() != 2 || wall_Z.ndim() != 2)
                throw std::runtime_error("wall_R and wall_Z must be 2-D");
            if (wall_R.shape(0) != wall_Z.shape(0) || wall_R.shape(1) != wall_Z.shape(1))
                throw std::runtime_error("wall_R and wall_Z shapes must match");
            if ((int)wall_phi_centers.size() != (int)wall_R.shape(0))
                throw std::runtime_error("wall_phi_centers length must equal wall_R.shape[0]");

            int N_seeds      = (int)R_seeds.size();
            int n_phi_wall   = (int)wall_R.shape(0);
            int n_theta_wall = (int)wall_R.shape(1);

            py::array_t<double> L_fwd({N_seeds}), L_bwd({N_seeds});
            py::array_t<double> R_hf({N_seeds}), Z_hf({N_seeds}), phi_hf({N_seeds});
            py::array_t<double> R_hb({N_seeds}), Z_hb({N_seeds}), phi_hb({N_seeds});
            py::array_t<int>    tt_fwd({N_seeds}), tt_bwd({N_seeds});

            cyna::trace_wall_hits_twall(
                buf(R_seeds,"R_seeds"), buf(Z_seeds,"Z_seeds"), N_seeds,
                phi_start, max_turns, DPhi,
                buf(BR,"BR"), buf(BPhi,"BPhi"), buf(BZ,"BZ"),
                buf(R_grid,"R_grid"), (int)R_grid.size(),
                buf(Z_grid,"Z_grid"), (int)Z_grid.size(),
                buf(Phi_grid,"Phi_grid"), (int)Phi_grid.size(),
                buf(wall_phi_centers,"wall_phi_centers"), n_phi_wall,
                buf(wall_R,"wall_R"), buf(wall_Z,"wall_Z"), n_theta_wall,
                n_threads,
                L_fwd.mutable_data(), L_bwd.mutable_data(),
                R_hf.mutable_data(), Z_hf.mutable_data(), phi_hf.mutable_data(),
                R_hb.mutable_data(), Z_hb.mutable_data(), phi_hb.mutable_data(),
                tt_fwd.mutable_data(), tt_bwd.mutable_data());

            return py::make_tuple(L_fwd, L_bwd,
                                  R_hf, Z_hf, phi_hf,
                                  R_hb, Z_hb, phi_hb,
                                  tt_fwd, tt_bwd);
        },
        py::arg("R_seeds"), py::arg("Z_seeds"), py::arg("phi_start"),
        py::arg("max_turns"), py::arg("DPhi"),
        py::arg("BR"), py::arg("BPhi"), py::arg("BZ"),
        py::arg("R_grid"), py::arg("Z_grid"), py::arg("Phi_grid"),
        py::arg("wall_phi_centers"), py::arg("wall_R"), py::arg("wall_Z"),
        py::arg("n_threads") = -1,
        "Like trace_connection_length_twall but also returns wall-hit coordinates.\n"
        "Returns (L_fwd, L_bwd, R_hit_fwd, Z_hit_fwd, phi_hit_fwd,\n"
        "                       R_hit_bwd, Z_hit_bwd, phi_hit_bwd,\n"
        "                       term_type_fwd, term_type_bwd).\n"
        "term_type: 0=no hit, 1=wall polygon, 2=field grid exit, 3=non-finite.\n"
        "NaN hit coords when term_type=0.");

    m.def("find_fixed_points_batch",
        [](py::array_t<double> R_seeds, py::array_t<double> Z_seeds,
           double phi_section, int n_turns, double DPhi,
           double fd_eps, int max_iter, double tol,
           py::array_t<double> BR, py::array_t<double> BPhi, py::array_t<double> BZ,
           py::array_t<double> R_grid, py::array_t<double> Z_grid,
           py::array_t<double> Phi_grid,
           int n_threads) -> py::tuple
        {
            if (n_threads <= 0) n_threads = (int)std::thread::hardware_concurrency();
            int N = (int)R_seeds.size();

            py::array_t<double> R_out({N}), Z_out({N}), res_out({N});
            py::array_t<int>    conv_out({N}), ptype_out({N});
            py::array_t<double> DPm_out({N, 4});
            py::array_t<double> eigr_out({N, 2}), eigi_out({N, 2});

            cyna::find_fixed_points_batch(
                buf(R_seeds,"R_seeds"), buf(Z_seeds,"Z_seeds"), N,
                phi_section, n_turns, DPhi, fd_eps, max_iter, tol,
                buf(BR,"BR"), buf(BPhi,"BPhi"), buf(BZ,"BZ"),
                buf(R_grid,"R_grid"), (int)R_grid.size(),
                buf(Z_grid,"Z_grid"), (int)Z_grid.size(),
                buf(Phi_grid,"Phi_grid"), (int)Phi_grid.size(),
                n_threads,
                R_out.mutable_data(), Z_out.mutable_data(),
                res_out.mutable_data(), conv_out.mutable_data(),
                DPm_out.mutable_data(),
                eigr_out.mutable_data(), eigi_out.mutable_data(),
                ptype_out.mutable_data());

            return py::make_tuple(R_out, Z_out, res_out, conv_out,
                                  DPm_out, eigr_out, eigi_out, ptype_out);
        },
        py::arg("R_seeds"), py::arg("Z_seeds"),
        py::arg("phi_section"), py::arg("n_turns"), py::arg("DPhi") = 0.05,
        py::arg("fd_eps") = 1e-4, py::arg("max_iter") = 40, py::arg("tol") = 1e-9,
        py::arg("BR"), py::arg("BPhi"), py::arg("BZ"),
        py::arg("R_grid"), py::arg("Z_grid"), py::arg("Phi_grid"),
        py::arg("n_threads") = -1,
        "Parallel Newton search for P^n fixed points (X/O points).\n"
        "Each seed is refined independently in its own thread.\n"
        "Returns (R_out, Z_out, residual, converged, DPm[N,4],\n"
        "         eig_r[N,2], eig_i[N,2], point_type[N]).\n"
        "point_type: 1=X-point (|Tr|>2), 0=O-point, -1=not converged.");

    m.def("compute_A_matrix_batch", &py_compute_A_matrix_batch,
        py::arg("R_arr"), py::arg("Z_arr"), py::arg("phi_arr"),
        py::arg("BR"), py::arg("BPhi"), py::arg("BZ"),
        py::arg("R_grid"), py::arg("Z_grid"), py::arg("Phi_grid"),
        py::arg("eps") = 1e-4,
        "Compute the 2x2 A-matrix at N orbit points using C++ trilinear interpolation.\n"
        "Returns ndarray of shape (N, 2, 2).\n"
        "A[k] = [[dg0/dR, dg0/dZ], [dg1/dR, dg1/dZ]] where g=[R*BR/BPhi, R*BZ/BPhi].");

    m.def("trace_surface_metrics_batch_twall", &py_trace_surface_metrics_batch_twall,
        py::arg("R_seeds"), py::arg("Z_seeds"),
        py::arg("R_axis"), py::arg("Z_axis"),
        py::arg("phi_start"), py::arg("N_turns"), py::arg("DPhi"),
        py::arg("BR"), py::arg("BPhi"), py::arg("BZ"),
        py::arg("R_grid"), py::arg("Z_grid"), py::arg("Phi_grid"),
        py::arg("wall_phi_centers"), py::arg("wall_R"), py::arg("wall_Z"),
        py::arg("fd_eps_R") = 1e-4, py::arg("fd_eps_Z") = 1e-4, py::arg("fd_eps_phi") = 1e-4,
        py::arg("n_threads") = -1,
        "Trace flux-surface seeds and return per-surface iota, B statistics and JxB metrics.");

    m.def("summarize_profile_objectives", &py_summarize_profile_objectives,
        py::arg("r_eff"), py::arg("iota"), py::arg("B_mean"), py::arg("B2_mean"),
        py::arg("B_min"), py::arg("B_max"), py::arg("JxB_mean"), py::arg("a_eff"),
        "Reduce per-surface metrics to fast Optuna objectives.");

    m.def("trace_poincare_beta_sweep", &py_trace_poincare_beta_sweep,
        py::arg("R_seeds"), py::arg("Z_seeds"),
        py::arg("phi_sections"),
        py::arg("N_turns"), py::arg("DPhi"),
        py::arg("BR"), py::arg("BPhi"), py::arg("BZ"),
        py::arg("R_grid"), py::arg("Z_grid"), py::arg("Phi_grid"),
        py::arg("wall_R"), py::arg("wall_Z"),
        py::arg("beta") = 0.0,
        py::arg("R_ax") = 0.0,
        py::arg("Z_ax") = 0.0,
        py::arg("a_eff") = 0.2,
        py::arg("alpha_pressure") = 2.0,
        py::arg("B_ref") = 1.0,
        py::arg("n_threads") = -1,
        "Trace Poincare sections for multiple seeds and multiple phi sections\n"
        "with on-the-fly beta field correction (diamagnetic + Pfirsch-Schluter).\n"
        "Returns (poi_R[N,n_sec,N_turns], poi_Z[N,n_sec,N_turns], poi_counts[N,n_sec]).\n"
        "NaN-padded where actual crossing count < N_turns.");

    m.def("trace_orbit_along_phi",
        [](double R0, double Z0, double phi0,
           double phi_span, double dphi_out, int n_turns_DPm,
           double DPhi, double fd_eps,
           py::array_t<double> BR, py::array_t<double> BPhi, py::array_t<double> BZ,
           py::array_t<double> R_grid, py::array_t<double> Z_grid,
           py::array_t<double> Phi_grid) -> py::tuple
        {
            int n_out = (int)std::ceil(phi_span / dphi_out) + 1;
            py::array_t<double> R_t({n_out}), Z_t({n_out}), phi_t({n_out});
            py::array_t<double> DPm_t({n_out, 4});
            py::array_t<int>    alive_t({n_out});

            cyna::trace_orbit_along_phi(
                R0, Z0, phi0, phi_span, dphi_out, n_turns_DPm, DPhi, fd_eps,
                buf(BR,"BR"), buf(BPhi,"BPhi"), buf(BZ,"BZ"),
                buf(R_grid,"R_grid"), (int)R_grid.size(),
                buf(Z_grid,"Z_grid"), (int)Z_grid.size(),
                buf(Phi_grid,"Phi_grid"), (int)Phi_grid.size(),
                n_out,
                R_t.mutable_data(), Z_t.mutable_data(), phi_t.mutable_data(),
                DPm_t.mutable_data(), alive_t.mutable_data());

            return py::make_tuple(R_t, Z_t, phi_t, DPm_t, alive_t);
        },
        py::arg("R0"), py::arg("Z0"), py::arg("phi0"),
        py::arg("phi_span"), py::arg("dphi_out"), py::arg("n_turns_DPm"),
        py::arg("DPhi") = 0.05, py::arg("fd_eps") = 1e-4,
        py::arg("BR"), py::arg("BPhi"), py::arg("BZ"),
        py::arg("R_grid"), py::arg("Z_grid"), py::arg("Phi_grid"),
        "Integrate field line from (R0,Z0,phi0) for phi_span radians, outputting\n"
        "(R,Z,phi,DPm[n,4],alive[n]) at evenly-spaced phi_out intervals.\n"
        "DPm is the P^n_turns_DPm Jacobian via central FD — used for ribbon visualization.");

    // -----------------------------------------------------------------------
    // coil_circular_field — analytic ring-coil B at a Cartesian point cloud
    // -----------------------------------------------------------------------
    m.def("coil_circular_field",
        [](double cx, double cy, double cz,
           double nx, double ny, double nz,
           double radius, double current,
           py::array_t<float> xyz_f32)
        -> py::tuple
        {
            if (xyz_f32.ndim() != 2 || xyz_f32.shape(1) != 3)
                throw std::runtime_error("xyz must have shape (N, 3)");
            if (!(xyz_f32.flags() & py::array::c_style))
                throw std::runtime_error("xyz must be C-contiguous");

            int N = (int)xyz_f32.shape(0);
            const float* pxyz = xyz_f32.data();

            py::array_t<float> BR({N}), BPhi({N}), BZ({N});
            float* pBR   = BR.mutable_data();
            float* pBPhi = BPhi.mutable_data();
            float* pBZ   = BZ.mutable_data();

#ifdef CYNA_CUDA_ENABLED
            bool ok = cyna_coil_circular_field_cuda(
                (float)cx,(float)cy,(float)cz,
                (float)nx,(float)ny,(float)nz,
                (float)radius,(float)current,
                pxyz, N, pBR, pBPhi, pBZ);
            if (!ok) {
                // CUDA failed — fall back to CPU
                cyna::circular_coil_field_cpu(
                    (float)cx,(float)cy,(float)cz,
                    (float)nx,(float)ny,(float)nz,
                    (float)radius,(float)current,
                    pxyz, N, pBR, pBPhi, pBZ);
            }
#else
            cyna::circular_coil_field_cpu(
                (float)cx,(float)cy,(float)cz,
                (float)nx,(float)ny,(float)nz,
                (float)radius,(float)current,
                pxyz, N, pBR, pBPhi, pBZ);
#endif
            return py::make_tuple(BR, BPhi, BZ);
        },
        py::arg("cx"),py::arg("cy"),py::arg("cz"),
        py::arg("nx"),py::arg("ny"),py::arg("nz"),
        py::arg("radius"),py::arg("current"),
        py::arg("xyz"),
        "Compute the magnetic field of one circular ring coil at N Cartesian\n"
        "field points.  Uses CUDA kernel when compiled with --with-cuda=y,\n"
        "otherwise falls back to OpenMP CPU implementation.\n"
        "Returns (BR, BPhi, BZ) — three (N,) float32 arrays in Tesla.");

    // -----------------------------------------------------------------------
    // coil_biot_savart — Biot-Savart for one arbitrary filamentary coil
    // -----------------------------------------------------------------------
    m.def("coil_biot_savart",
        [](py::array_t<float> seg_starts,
           py::array_t<float> seg_ends,
           double current,
           py::array_t<float> xyz_f32)
        -> py::tuple
        {
            if (seg_starts.ndim()!=2 || seg_starts.shape(1)!=3)
                throw std::runtime_error("seg_starts must have shape (N_seg, 3)");
            if (seg_ends.ndim()!=2 || seg_ends.shape(1)!=3)
                throw std::runtime_error("seg_ends must have shape (N_seg, 3)");
            if (xyz_f32.ndim()!=2 || xyz_f32.shape(1)!=3)
                throw std::runtime_error("xyz must have shape (N, 3)");
            if (!(seg_starts.flags() & py::array::c_style) ||
                !(seg_ends.flags()   & py::array::c_style) ||
                !(xyz_f32.flags()    & py::array::c_style))
                throw std::runtime_error("All arrays must be C-contiguous");

            int N_seg = (int)seg_starts.shape(0);
            int N     = (int)xyz_f32.shape(0);
            const float* pss  = seg_starts.data();
            const float* pse  = seg_ends.data();
            const float* pxyz = xyz_f32.data();

            py::array_t<float> BR({N}), BPhi({N}), BZ({N});
            float* pBR   = BR.mutable_data();
            float* pBPhi = BPhi.mutable_data();
            float* pBZ   = BZ.mutable_data();

#ifdef CYNA_CUDA_ENABLED
            bool ok = cyna_coil_biot_savart_cuda(
                pss, pse, N_seg, (float)current,
                pxyz, N, pBR, pBPhi, pBZ);
            if (!ok) {
                cyna::biot_savart_field_cpu(pss, pse, N_seg, (float)current,
                                            pxyz, N, pBR, pBPhi, pBZ);
            }
#else
            cyna::biot_savart_field_cpu(pss, pse, N_seg, (float)current,
                                        pxyz, N, pBR, pBPhi, pBZ);
#endif
            return py::make_tuple(BR, BPhi, BZ);
        },
        py::arg("seg_starts"),py::arg("seg_ends"),py::arg("current"),py::arg("xyz"),
        "Compute Biot-Savart field for one arbitrary filamentary coil.\n"
        "seg_starts/seg_ends: (N_seg,3) float32 Cartesian segment endpoints (m).\n"
        "Returns (BR, BPhi, BZ) — three (N,) float32 arrays in Tesla.\n"
        "Uses CUDA kernel when compiled with --with-cuda=y.");
}
