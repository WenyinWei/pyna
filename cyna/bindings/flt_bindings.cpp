// flt_bindings.cpp — pybind11 bindings for cyna Poincaré tracer + coil fields
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <thread>
#include <stdexcept>
#include <cmath>
#include <limits>
#include <vector>
#include <string>
#include <filesystem>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#include "cyna/poincare.hpp"
#include "cyna/objectives.hpp"
#include "cyna/coil_field.hpp"

namespace py = pybind11;

namespace {

using cuda_circular_fn = bool (*)(
    float,float,float, float,float,float, float,float,
    const float*,int, float*,float*,float*);
using cuda_biot_fn = bool (*)(
    const float*,const float*,int,float,
    const float*,int, float*,float*,float*);

struct CudaBackend {
    bool attempted = false;
    void* handle = nullptr;
    cuda_circular_fn circular = nullptr;
    cuda_biot_fn biot = nullptr;
};

static std::filesystem::path _current_module_dir() {
#ifdef _WIN32
    HMODULE module = nullptr;
    char path[MAX_PATH] = {0};
    if (GetModuleHandleExA(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
            GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCSTR>(&_current_module_dir),
            &module) &&
        GetModuleFileNameA(module, path, MAX_PATH) > 0) {
        return std::filesystem::path(path).parent_path();
    }
#else
    Dl_info info;
    if (dladdr(reinterpret_cast<void*>(&_current_module_dir), &info) && info.dli_fname) {
        return std::filesystem::path(info.dli_fname).parent_path();
    }
#endif
    return std::filesystem::current_path();
}

static void* _open_cuda_backend(const std::filesystem::path& path) {
#ifdef _WIN32
    return reinterpret_cast<void*>(LoadLibraryA(path.string().c_str()));
#else
    return dlopen(path.string().c_str(), RTLD_LAZY | RTLD_LOCAL);
#endif
}

static void* _cuda_symbol(void* handle, const char* name) {
#ifdef _WIN32
    return reinterpret_cast<void*>(
        GetProcAddress(reinterpret_cast<HMODULE>(handle), name));
#else
    return dlsym(handle, name);
#endif
}

static CudaBackend& _cuda_backend() {
    static CudaBackend backend;
    if (backend.attempted) {
        return backend;
    }
    backend.attempted = true;

    const auto dir = _current_module_dir();
#ifdef _WIN32
    const std::vector<std::string> names = {"_cyna_cuda_backend.dll"};
#elif defined(__APPLE__)
    const std::vector<std::string> names = {
        "_cyna_cuda_backend.so", "_cyna_cuda_backend.dylib"};
#else
    const std::vector<std::string> names = {"_cyna_cuda_backend.so"};
#endif
    for (const auto& name : names) {
        auto candidate = dir / name;
        if (!std::filesystem::exists(candidate)) {
            continue;
        }
        void* handle = _open_cuda_backend(candidate);
        if (!handle) {
            continue;
        }
        auto circular = reinterpret_cast<cuda_circular_fn>(
            _cuda_symbol(handle, "cyna_coil_circular_field_cuda"));
        auto biot = reinterpret_cast<cuda_biot_fn>(
            _cuda_symbol(handle, "cyna_coil_biot_savart_cuda"));
        if (circular && biot) {
            backend.handle = handle;
            backend.circular = circular;
            backend.biot = biot;
            break;
        }
    }
    return backend;
}

static bool _cuda_backend_available() {
    auto& backend = _cuda_backend();
    return backend.circular != nullptr && backend.biot != nullptr;
}

}  // namespace

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
    py::array_t<double> BR, py::array_t<double> BZ, py::array_t<double> BPhi,
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

// ---------------------------------------------------------------------------
// progress_DX_pol_along_orbit
// ---------------------------------------------------------------------------
static py::array_t<double> py_progress_DX_pol_along_orbit(
    py::array_t<double> R_traj,
    py::array_t<double> Z_traj,
    py::array_t<double> phi_traj,
    py::array_t<double> BR, py::array_t<double> BZ, py::array_t<double> BPhi,
    py::array_t<double> R_grid,
    py::array_t<double> Z_grid,
    py::array_t<double> Phi_grid,
    double max_step)
{
    const int n_pts = (int)R_traj.size();
    if ((int)Z_traj.size() != n_pts || (int)phi_traj.size() != n_pts) {
        throw std::runtime_error("R_traj, Z_traj and phi_traj must have the same length");
    }
    if (max_step <= 0.0 || !std::isfinite(max_step)) {
        throw std::runtime_error("max_step must be finite and positive");
    }

    py::array_t<double> out({n_pts, 2, 2});
    cyna::progress_DX_pol_along_orbit(
        buf(R_traj, "R_traj"), buf(Z_traj, "Z_traj"), buf(phi_traj, "phi_traj"),
        n_pts,
        out.mutable_data(),
        buf(BR,"BR"), buf(BZ,"BZ"), buf(BPhi,"BPhi"),
        buf(R_grid,"R_grid"), (int)R_grid.size(),
        buf(Z_grid,"Z_grid"), (int)Z_grid.size(),
        buf(Phi_grid,"Phi_grid"), (int)Phi_grid.size(),
        max_step);
    return out;
}

// ---------------------------------------------------------------------------
// progress_delta_X_along_orbit
// ---------------------------------------------------------------------------
static py::array_t<double> py_progress_delta_X_along_orbit(
    py::array_t<double> R_traj,
    py::array_t<double> Z_traj,
    py::array_t<double> phi_traj,
    py::array_t<double> delta_X0,
    py::array_t<double> BR, py::array_t<double> BZ, py::array_t<double> BPhi,
    py::array_t<double> dBR, py::array_t<double> dBZ, py::array_t<double> dBPhi,
    py::array_t<double> R_grid,
    py::array_t<double> Z_grid,
    py::array_t<double> Phi_grid,
    double max_step)
{
    const int n_pts = (int)R_traj.size();
    if ((int)Z_traj.size() != n_pts || (int)phi_traj.size() != n_pts) {
        throw std::runtime_error("R_traj, Z_traj and phi_traj must have the same length");
    }
    if ((int)delta_X0.size() != 2) {
        throw std::runtime_error("delta_X0 must have length 2");
    }
    if (max_step <= 0.0 || !std::isfinite(max_step)) {
        throw std::runtime_error("max_step must be finite and positive");
    }

    py::array_t<double> out({n_pts, 2});
    cyna::progress_delta_X_along_orbit(
        buf(R_traj, "R_traj"), buf(Z_traj, "Z_traj"), buf(phi_traj, "phi_traj"),
        n_pts,
        buf(delta_X0, "delta_X0"),
        out.mutable_data(),
        buf(BR,"BR"), buf(BZ,"BZ"), buf(BPhi,"BPhi"),
        buf(dBR,"dBR"), buf(dBZ,"dBZ"), buf(dBPhi,"dBPhi"),
        buf(R_grid,"R_grid"), (int)R_grid.size(),
        buf(Z_grid,"Z_grid"), (int)Z_grid.size(),
        buf(Phi_grid,"Phi_grid"), (int)Phi_grid.size(),
        max_step);
	    return out;
	}

// ---------------------------------------------------------------------------
// evolve_delta_X_cycle_along_cycle
// ---------------------------------------------------------------------------
static py::array_t<double> py_evolve_delta_X_cycle_along_cycle(
    py::array_t<double> R_traj,
    py::array_t<double> Z_traj,
    py::array_t<double> phi_traj,
    py::array_t<double> delta_X0,
    py::array_t<double> BR, py::array_t<double> BZ, py::array_t<double> BPhi,
    py::array_t<double> dBR, py::array_t<double> dBZ, py::array_t<double> dBPhi,
    py::array_t<double> R_grid,
    py::array_t<double> Z_grid,
    py::array_t<double> Phi_grid,
    double max_step)
{
    const int n_pts = (int)R_traj.size();
    if ((int)Z_traj.size() != n_pts || (int)phi_traj.size() != n_pts) {
        throw std::runtime_error("R_traj, Z_traj and phi_traj must have the same length");
    }
    if ((int)delta_X0.size() != 2) {
        throw std::runtime_error("delta_X0 must have length 2");
    }
    if (max_step <= 0.0 || !std::isfinite(max_step)) {
        throw std::runtime_error("max_step must be finite and positive");
    }

    py::array_t<double> out({n_pts, 2});
    cyna::evolve_delta_X_cycle_along_cycle(
        buf(R_traj, "R_traj"), buf(Z_traj, "Z_traj"), buf(phi_traj, "phi_traj"),
        n_pts,
        buf(delta_X0, "delta_X0"),
        out.mutable_data(),
        buf(BR,"BR"), buf(BZ,"BZ"), buf(BPhi,"BPhi"),
        buf(dBR,"dBR"), buf(dBZ,"dBZ"), buf(dBPhi,"dBPhi"),
        buf(R_grid,"R_grid"), (int)R_grid.size(),
        buf(Z_grid,"Z_grid"), (int)Z_grid.size(),
        buf(Phi_grid,"Phi_grid"), (int)Phi_grid.size(),
        max_step);
    return out;
}

static py::tuple py_trace_poincare_batch(
    py::array_t<double> R_seeds,
    py::array_t<double> Z_seeds,
    double phi_section,
    int N_turns,
    double DPhi,
    py::array_t<double> BR, py::array_t<double> BZ, py::array_t<double> BPhi,
    py::array_t<double> R_grid,
    py::array_t<double> Z_grid,
    py::array_t<double> Phi_grid,
    py::array_t<double> wall_R,
    py::array_t<double> wall_Z,
    int n_threads,
    int direction)
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
        buf(BR,"BR"), buf(BZ,"BZ"), buf(BPhi,"BPhi"),
        buf(R_grid, "R_grid"), nR,
        buf(Z_grid, "Z_grid"), nZ,
        buf(Phi_grid, "Phi_grid"), nPhi,
        buf(wall_R, "wall_R"), buf(wall_Z, "wall_Z"), n_wall,
        n_threads,
        poi_counts.mutable_data(),
        poi_R_flat.mutable_data(),
        poi_Z_flat.mutable_data(),
        direction);

    return py::make_tuple(poi_counts, poi_R_flat, poi_Z_flat);
}

static py::tuple py_trace_poincare_batch_twall(
    py::array_t<double> R_seeds,
    py::array_t<double> Z_seeds,
    double phi_section,
    int N_turns,
    double DPhi,
    py::array_t<double> BR, py::array_t<double> BZ, py::array_t<double> BPhi,
    py::array_t<double> R_grid,
    py::array_t<double> Z_grid,
    py::array_t<double> Phi_grid,
    py::array_t<double> wall_phi_centers,
    py::array_t<double> wall_R,
    py::array_t<double> wall_Z,
    int n_threads,
    int direction)
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
        buf(BR,"BR"), buf(BZ,"BZ"), buf(BPhi,"BPhi"),
        buf(R_grid, "R_grid"), nR,
        buf(Z_grid, "Z_grid"), nZ,
        buf(Phi_grid, "Phi_grid"), nPhi,
        buf(wall_phi_centers, "wall_phi_centers"), n_phi_wall,
        buf(wall_R, "wall_R"), buf(wall_Z, "wall_Z"), n_theta_wall,
        n_threads,
        poi_counts.mutable_data(),
        poi_R_flat.mutable_data(),
        poi_Z_flat.mutable_data(),
        direction);

    return py::make_tuple(poi_counts, poi_R_flat, poi_Z_flat);
}

static py::tuple py_trace_poincare_multi(
    py::array_t<double> R_seeds,
    py::array_t<double> Z_seeds,
    py::array_t<double> phi_sections,
    int N_turns,
    double DPhi,
    py::array_t<double> BR, py::array_t<double> BZ, py::array_t<double> BPhi,
    py::array_t<double> R_grid,
    py::array_t<double> Z_grid,
    py::array_t<double> Phi_grid,
    py::array_t<double> wall_R,
    py::array_t<double> wall_Z,
    int n_threads,
    int direction)
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
                buf(BR,"BR"), buf(BZ,"BZ"), buf(BPhi,"BPhi"),
                buf(R_grid, "R_grid"), nR,
                buf(Z_grid, "Z_grid"), nZ,
                buf(Phi_grid, "Phi_grid"), nPhi,
                buf(wall_R, "wall_R"), buf(wall_Z, "wall_Z"), n_wall,
                poi_counts.mutable_data(),
                poi_R_flat.mutable_data(),
                poi_Z_flat.mutable_data(),
                direction);
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
    py::array_t<double> BR, py::array_t<double> BZ, py::array_t<double> BPhi,
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
        buf(BR,"BR"), buf(BZ,"BZ"), buf(BPhi,"BPhi"),
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

static py::tuple py_compute_cycle_perturbation_shift(
    double R0, double Z0, double phi0,
    double phi_span, double dphi_out, double DPhi, double fd_eps,
    py::array_t<double> BR_base, py::array_t<double> BZ_base, py::array_t<double> BPhi_base,
    py::array_t<double> BR_pert, py::array_t<double> BZ_pert, py::array_t<double> BPhi_pert,
    py::array_t<double> R_grid,
    py::array_t<double> Z_grid,
    py::array_t<double> Phi_grid)
{
    const int nR = (int)R_grid.size();
    const int nZ = (int)Z_grid.size();
    const int nPhi = (int)Phi_grid.size();
    const int n_out = (int)std::ceil(std::abs(phi_span) / std::abs(dphi_out)) + 1;
    const double* Rg = buf(R_grid, "R_grid");
    const double* Zg = buf(Z_grid, "Z_grid");
    const double* Pg = buf(Phi_grid, "Phi_grid");
    const double* br0 = buf(BR_base, "BR_base");
    const double* bz0 = buf(BZ_base, "BZ_base");
    const double* bp0 = buf(BPhi_base, "BPhi_base");
    const double* br1 = buf(BR_pert, "BR_pert");
    const double* bz1 = buf(BZ_pert, "BZ_pert");
    const double* bp1 = buf(BPhi_pert, "BPhi_pert");

    py::array_t<double> R_t({n_out}), Z_t({n_out}), phi_t({n_out});
    py::array_t<double> DP_t({n_out, 4});
    py::array_t<double> dXpol_t({n_out, 2});
    py::array_t<double> dXcyc_t({n_out, 2});
    py::array_t<int> alive_t({n_out});

    auto f_eval = [&](const double* BR, const double* BZ, const double* BPhi,
                      double R, double Z, double phi, double& fR, double& fZ) {
        double bp = cyna::interp3d(BPhi, Rg, nR, Zg, nZ, Pg, nPhi, R, Z, phi);
        double br = cyna::interp3d(BR,   Rg, nR, Zg, nZ, Pg, nPhi, R, Z, phi);
        double bz = cyna::interp3d(BZ,   Rg, nR, Zg, nZ, Pg, nPhi, R, Z, phi);
        if (!std::isfinite(bp) || std::abs(bp) < 1e-30 ||
            !std::isfinite(br) || !std::isfinite(bz)) {
            fR = std::numeric_limits<double>::quiet_NaN();
            fZ = std::numeric_limits<double>::quiet_NaN();
            return;
        }
        fR = R * br / bp;
        fZ = R * bz / bp;
    };

    auto A_eval = [&](double R, double Z, double phi, double A[4]) {
        double BRv, dBR_dR, dBR_dZ;
        double BZv, dBZ_dR, dBZ_dZ;
        double BPv, dBP_dR, dBP_dZ;
        if (!cyna::interp3d_grad(BRv, dBR_dR, dBR_dZ, br0, Rg,nR,Zg,nZ,Pg,nPhi, R,Z,phi) ||
            !cyna::interp3d_grad(BZv, dBZ_dR, dBZ_dZ, bz0, Rg,nR,Zg,nZ,Pg,nPhi, R,Z,phi) ||
            !cyna::interp3d_grad(BPv, dBP_dR, dBP_dZ, bp0, Rg,nR,Zg,nZ,Pg,nPhi, R,Z,phi) ||
            std::abs(BPv) < 1e-30) {
            A[0] = A[1] = A[2] = A[3] = std::numeric_limits<double>::quiet_NaN();
            return;
        }
        const double invBp = 1.0 / BPv;
        const double invBp2 = invBp / BPv;
        A[0] = BRv * invBp + R * (dBR_dR * invBp - BRv * invBp2 * dBP_dR);
        A[1] = R * (dBR_dZ * invBp - BRv * invBp2 * dBP_dZ);
        A[2] = BZv * invBp + R * (dBZ_dR * invBp - BZv * invBp2 * dBP_dR);
        A[3] = R * (dBZ_dZ * invBp - BZv * invBp2 * dBP_dZ);
    };

    auto delta_f_eval = [&](double R, double Z, double phi, double& dF0, double& dF1) {
        double bp_base = cyna::interp3d(bp0, Rg, nR, Zg, nZ, Pg, nPhi, R, Z, phi);
        double br_base = cyna::interp3d(br0, Rg, nR, Zg, nZ, Pg, nPhi, R, Z, phi);
        double bz_base = cyna::interp3d(bz0, Rg, nR, Zg, nZ, Pg, nPhi, R, Z, phi);
        double bp_pert = cyna::interp3d(bp1, Rg, nR, Zg, nZ, Pg, nPhi, R, Z, phi);
        double br_pert = cyna::interp3d(br1, Rg, nR, Zg, nZ, Pg, nPhi, R, Z, phi);
        double bz_pert = cyna::interp3d(bz1, Rg, nR, Zg, nZ, Pg, nPhi, R, Z, phi);
        if (!std::isfinite(bp_base) || std::abs(bp_base) < 1e-30 ||
            !std::isfinite(br_base) || !std::isfinite(bz_base) ||
            !std::isfinite(bp_pert) || !std::isfinite(br_pert) || !std::isfinite(bz_pert)) {
            dF0 = std::numeric_limits<double>::quiet_NaN();
            dF1 = std::numeric_limits<double>::quiet_NaN();
            return;
        }
        const double dBRv = br_pert - br_base;
        const double dBZv = bz_pert - bz_base;
        const double dBPv = bp_pert - bp_base;
        const double invBp2 = 1.0 / (bp_base * bp_base);
        dF0 = R * (dBRv * bp_base - br_base * dBPv) * invBp2;
        dF1 = R * (dBZv * bp_base - bz_base * dBPv) * invBp2;
    };

    auto rhs = [&](const double y[8], double phi, double out[8]) {
        double fR0, fZ0, dF0, dF1;
        f_eval(br0, bz0, bp0, y[0], y[1], phi, fR0, fZ0);
        delta_f_eval(y[0], y[1], phi, dF0, dF1);
        double A[4];
        A_eval(y[0], y[1], phi, A);
        out[0] = fR0;
        out[1] = fZ0;
        // M is row-major [[m00,m01],[m10,m11]]
        out[2] = A[0] * y[2] + A[1] * y[4];
        out[3] = A[0] * y[3] + A[1] * y[5];
        out[4] = A[2] * y[2] + A[3] * y[4];
        out[5] = A[2] * y[3] + A[3] * y[5];
        out[6] = A[0] * y[6] + A[1] * y[7] + dF0;
        out[7] = A[2] * y[6] + A[3] * y[7] + dF1;
    };

    auto rk4 = [&](double y[8], double phi, double h) {
        double k1[8], k2[8], k3[8], k4[8], yt[8];
        rhs(y, phi, k1);
        for (int i=0;i<8;i++) yt[i] = y[i] + 0.5 * h * k1[i];
        rhs(yt, phi + 0.5*h, k2);
        for (int i=0;i<8;i++) yt[i] = y[i] + 0.5 * h * k2[i];
        rhs(yt, phi + 0.5*h, k3);
        for (int i=0;i<8;i++) yt[i] = y[i] + h * k3[i];
        rhs(yt, phi + h, k4);
        for (int i=0;i<8;i++) y[i] += (h / 6.0) * (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]);
    };

    double y[8] = {R0, Z0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0};
    double phi = phi0;
    const double phi_end = phi0 + phi_span;
    const double dir = (phi_span >= 0.0) ? 1.0 : -1.0;
    int out_idx = 0;
    double next_out = phi0;

    auto record = [&]() {
        if (out_idx >= n_out) return;
        R_t.mutable_data()[out_idx] = y[0];
        Z_t.mutable_data()[out_idx] = y[1];
        phi_t.mutable_data()[out_idx] = next_out;
        for (int k=0;k<4;k++) DP_t.mutable_data()[4*out_idx+k] = y[2+k];
        dXpol_t.mutable_data()[2*out_idx+0] = y[6];
        dXpol_t.mutable_data()[2*out_idx+1] = y[7];
        alive_t.mutable_data()[out_idx] = (std::isfinite(y[0]) && std::isfinite(y[1])) ? 1 : 0;
        out_idx++;
    };

    record();
    next_out += dphi_out;
    while ((dir > 0 ? phi < phi_end - 1e-13 : phi > phi_end + 1e-13) && out_idx < n_out) {
        double target = next_out;
        if (dir > 0) target = std::min(target, phi_end);
        else target = std::max(target, phi_end);
        while (dir > 0 ? phi < target - 1e-13 : phi > target + 1e-13) {
            double h = dir * std::min(std::abs(DPhi), std::abs(target - phi));
            rk4(y, phi, h);
            phi += h;
            if (!std::isfinite(y[0]) || !std::isfinite(y[1])) break;
        }
        record();
        next_out += dphi_out;
    }

    // Solve (I-DPm) dx0 = dXpol_end for periodic cycle response.
    const double m00 = y[2], m01 = y[3], m10 = y[4], m11 = y[5];
    const double a00 = 1.0 - m00, a01 = -m01, a10 = -m10, a11 = 1.0 - m11;
    const double det = a00*a11 - a01*a10;
    double dx0 = std::numeric_limits<double>::quiet_NaN();
    double dz0 = std::numeric_limits<double>::quiet_NaN();
    if (std::abs(det) > 1e-30) {
        dx0 = ( a11*y[6] - a01*y[7]) / det;
        dz0 = (-a10*y[6] + a00*y[7]) / det;
    } else if (std::hypot(y[6], y[7]) < 1e-30) {
        dx0 = 0.0;
        dz0 = 0.0;
    }

    for (int i=0; i<n_out; ++i) {
        double* M = DP_t.mutable_data() + 4*i;
        double* xp = dXpol_t.mutable_data() + 2*i;
        dXcyc_t.mutable_data()[2*i+0] = xp[0] + M[0]*dx0 + M[1]*dz0;
        dXcyc_t.mutable_data()[2*i+1] = xp[1] + M[2]*dx0 + M[3]*dz0;
    }

    py::array_t<double> dx0_arr({2});
    dx0_arr.mutable_data()[0] = dx0;
    dx0_arr.mutable_data()[1] = dz0;
    return py::make_tuple(R_t, Z_t, phi_t, DP_t, dXpol_t, dXcyc_t, dx0_arr, alive_t);
}

static py::tuple py_trace_poincare_dpk_growth(
    double R0, double Z0, double phi0,
    int max_returns,
    double return_period,
    int record_stride,
    double DPhi,
    py::array_t<double> BR, py::array_t<double> BZ, py::array_t<double> BPhi,
    py::array_t<double> R_grid,
    py::array_t<double> Z_grid,
    py::array_t<double> Phi_grid)
{
    if (max_returns < 0)
        throw std::runtime_error("max_returns must be non-negative");
    if (record_stride <= 0)
        throw std::runtime_error("record_stride must be positive");
    if (!std::isfinite(return_period) || std::abs(return_period) <= 1e-14)
        throw std::runtime_error("return_period must be finite and non-zero");
    if (!std::isfinite(DPhi) || std::abs(DPhi) <= 1e-14)
        throw std::runtime_error("DPhi must be finite and non-zero");

    const int n_out = (max_returns + record_stride - 1) / record_stride;
    py::array_t<int> k_t({n_out});
    py::array_t<double> R_t({n_out}), Z_t({n_out}), phi_t({n_out});
    py::array_t<double> DPk_t({n_out, 4});
    py::array_t<double> eig_abs_t({n_out, 2});
    py::array_t<int> alive_t({n_out});

    cyna::trace_poincare_dpk_growth(
        R0, Z0, phi0, max_returns, return_period, record_stride, DPhi,
        buf(BR,"BR"), buf(BZ,"BZ"), buf(BPhi,"BPhi"),
        buf(R_grid,"R_grid"), (int)R_grid.size(),
        buf(Z_grid,"Z_grid"), (int)Z_grid.size(),
        buf(Phi_grid,"Phi_grid"), (int)Phi_grid.size(),
        n_out,
        k_t.mutable_data(),
        R_t.mutable_data(), Z_t.mutable_data(), phi_t.mutable_data(),
        DPk_t.mutable_data(), eig_abs_t.mutable_data(), alive_t.mutable_data());

    return py::make_tuple(k_t, R_t, Z_t, phi_t, DPk_t, eig_abs_t, alive_t);
}

static py::tuple py_trace_poincare_dpk_growth_twall(
    double R0, double Z0, double phi0,
    int max_returns,
    double return_period,
    int record_stride,
    double DPhi,
    py::array_t<double> BR, py::array_t<double> BZ, py::array_t<double> BPhi,
    py::array_t<double> R_grid,
    py::array_t<double> Z_grid,
    py::array_t<double> Phi_grid,
    py::array_t<double> wall_phi_centers,
    py::array_t<double> wall_R,
    py::array_t<double> wall_Z,
    bool stop_at_wall)
{
    if (max_returns < 0)
        throw std::runtime_error("max_returns must be non-negative");
    if (record_stride <= 0)
        throw std::runtime_error("record_stride must be positive");
    if (!std::isfinite(return_period) || std::abs(return_period) <= 1e-14)
        throw std::runtime_error("return_period must be finite and non-zero");
    if (!std::isfinite(DPhi) || std::abs(DPhi) <= 1e-14)
        throw std::runtime_error("DPhi must be finite and non-zero");
    if (wall_R.ndim() != 2 || wall_Z.ndim() != 2)
        throw std::runtime_error("wall_R and wall_Z must be 2-D arrays [n_phi_wall, n_theta_wall]");
    if (wall_R.shape(0) != wall_Z.shape(0) || wall_R.shape(1) != wall_Z.shape(1))
        throw std::runtime_error("wall_R and wall_Z shapes must match");
    if ((int)wall_phi_centers.size() != (int)wall_R.shape(0))
        throw std::runtime_error("wall_phi_centers length must equal wall_R.shape[0]");

    const int n_out = (max_returns + record_stride - 1) / record_stride;
    py::array_t<int> k_t({n_out});
    py::array_t<double> R_t({n_out}), Z_t({n_out}), phi_t({n_out});
    py::array_t<double> DPk_t({n_out, 4});
    py::array_t<double> eig_abs_t({n_out, 2});
    py::array_t<int> alive_t({n_out});
    py::array_t<double> hit_t({4});
    py::array_t<int> term_t({1});

    cyna::trace_poincare_dpk_growth_twall(
        R0, Z0, phi0, max_returns, return_period, record_stride, DPhi,
        buf(BR,"BR"), buf(BZ,"BZ"), buf(BPhi,"BPhi"),
        buf(R_grid,"R_grid"), (int)R_grid.size(),
        buf(Z_grid,"Z_grid"), (int)Z_grid.size(),
        buf(Phi_grid,"Phi_grid"), (int)Phi_grid.size(),
        buf(wall_phi_centers, "wall_phi_centers"),
        buf(wall_R, "wall_R"), buf(wall_Z, "wall_Z"),
        (int)wall_R.shape(0), (int)wall_R.shape(1),
        stop_at_wall,
        n_out,
        k_t.mutable_data(),
        R_t.mutable_data(), Z_t.mutable_data(), phi_t.mutable_data(),
        DPk_t.mutable_data(), eig_abs_t.mutable_data(), alive_t.mutable_data(),
        hit_t.mutable_data(), term_t.mutable_data());

    return py::make_tuple(k_t, R_t, Z_t, phi_t, DPk_t, eig_abs_t, alive_t, hit_t, term_t);
}

static py::tuple py_trace_poincare_beta_sweep(
    py::array_t<double> R_seeds,
    py::array_t<double> Z_seeds,
    py::array_t<double> phi_sections,
    int N_turns,
    double DPhi,
    py::array_t<double> BR, py::array_t<double> BZ, py::array_t<double> BPhi,
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
        buf(BR,"BR"), buf(BZ,"BZ"), buf(BPhi,"BPhi"),
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
                               py::array_t<double> BR, py::array_t<double> BZ, py::array_t<double> BPhi,
                               py::array_t<double> Rg, py::array_t<double> Zg,
                               py::array_t<double> Pg) {
        cyna::rk4_step(R, Z, phi, DPhi,
            BR.data(), BZ.data(), BPhi.data(),
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
	        py::arg("BR"), py::arg("BZ"), py::arg("BPhi"),
	        py::arg("R_grid"), py::arg("Z_grid"), py::arg("Phi_grid"),
	        py::arg("wall_R"), py::arg("wall_Z"),
	        py::arg("n_threads") = -1,
	        py::arg("direction") = +1,
	        "Trace Poincaré section for multiple seeds against a fixed 2-D wall slice.\n"
	        "direction=+1 traces phi-increasing; direction=-1 traces phi-decreasing.\n"
	        "Returns (poi_counts, poi_R_flat, poi_Z_flat).");

	    m.def("trace_poincare_batch_twall", &py_trace_poincare_batch_twall,
	        py::arg("R_seeds"), py::arg("Z_seeds"), py::arg("phi_section"),
	        py::arg("N_turns"), py::arg("DPhi"),
	        py::arg("BR"), py::arg("BZ"), py::arg("BPhi"),
	        py::arg("R_grid"), py::arg("Z_grid"), py::arg("Phi_grid"),
	        py::arg("wall_phi_centers"), py::arg("wall_R"), py::arg("wall_Z"),
	        py::arg("n_threads") = -1,
	        py::arg("direction") = +1,
	        "Trace Poincaré section for multiple seeds against a toroidally varying wall.\n"
	        "direction=+1 traces phi-increasing; direction=-1 traces phi-decreasing.\n"
	        "Returns (poi_counts, poi_R_flat, poi_Z_flat).");

	    m.def("trace_poincare_multi", &py_trace_poincare_multi,
	        py::arg("R_seeds"), py::arg("Z_seeds"), py::arg("phi_sections"),
	        py::arg("N_turns"), py::arg("DPhi"),
	        py::arg("BR"), py::arg("BZ"), py::arg("BPhi"),
	        py::arg("R_grid"), py::arg("Z_grid"), py::arg("Phi_grid"),
	        py::arg("wall_R"), py::arg("wall_Z"),
	        py::arg("n_threads") = -1,
	        py::arg("direction") = +1,
	        "Trace Poincaré sections for multiple seeds and multiple phi sections.\n"
	        "direction=+1 traces phi-increasing; direction=-1 traces phi-decreasing.\n"
	        "Returns (poi_counts [N_seeds x n_sec], poi_R_flat, poi_Z_flat).");

    m.def("trace_connection_length_twall",
        [](py::array_t<double> R_seeds, py::array_t<double> Z_seeds,
           double phi_start, int max_turns, double DPhi,
           py::array_t<double> BR, py::array_t<double> BZ, py::array_t<double> BPhi,
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
                buf(BR,"BR"), buf(BZ,"BZ"), buf(BPhi,"BPhi"),
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
        py::arg("BR"), py::arg("BZ"), py::arg("BPhi"),
        py::arg("R_grid"), py::arg("Z_grid"), py::arg("Phi_grid"),
        py::arg("wall_phi_centers"), py::arg("wall_R"), py::arg("wall_Z"),
        py::arg("n_threads") = -1,
        "Compute connection length (forward + backward) for each seed against toroidal wall.\n"
        "Returns (L_fwd, L_bwd) in metres; sentinel=1e30 means no termination within max_turns.");

    m.def("trace_wall_hits_twall",
        [](py::array_t<double> R_seeds, py::array_t<double> Z_seeds,
           double phi_start, int max_turns, double DPhi,
           py::array_t<double> BR, py::array_t<double> BZ, py::array_t<double> BPhi,
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
                buf(BR,"BR"), buf(BZ,"BZ"), buf(BPhi,"BPhi"),
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
        py::arg("BR"), py::arg("BZ"), py::arg("BPhi"),
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
           double phi_section, int m_turns, double DPhi,
           double fd_eps, int max_iter, double tol,
           py::array_t<double> BR, py::array_t<double> BZ, py::array_t<double> BPhi,
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
                phi_section, m_turns, DPhi, fd_eps, max_iter, tol,
                buf(BR,"BR"), buf(BZ,"BZ"), buf(BPhi,"BPhi"),
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
        py::arg("phi_section"), py::arg("m_turns"), py::arg("DPhi") = 0.05,
        py::arg("fd_eps") = 1e-4, py::arg("max_iter") = 40, py::arg("tol") = 1e-9,
        py::arg("BR"), py::arg("BZ"), py::arg("BPhi"),
        py::arg("R_grid"), py::arg("Z_grid"), py::arg("Phi_grid"),
        py::arg("n_threads") = -1,
        "Parallel Newton search for P^n fixed points (X/O points).\n"
        "Each seed is refined independently in its own thread.\n"
        "Returns (R_out, Z_out, residual, converged, DPm[N,4],\n"
        "         eig_r[N,2], eig_i[N,2], point_type[N]).\n"
        "point_type: 1=X-point (|Tr|>2), 0=O-point, -1=not converged.");

    m.def("compute_A_matrix_batch", &py_compute_A_matrix_batch,
        py::arg("R_arr"), py::arg("Z_arr"), py::arg("phi_arr"),
        py::arg("BR"), py::arg("BZ"), py::arg("BPhi"),
        py::arg("R_grid"), py::arg("Z_grid"), py::arg("Phi_grid"),
        py::arg("eps") = 1e-4,
        "Compute the 2x2 A-matrix at N orbit points using C++ trilinear interpolation.\n"
        "Returns ndarray of shape (N, 2, 2).\n"
        "A[k] = [[dg0/dR, dg0/dZ], [dg1/dR, dg1/dZ]] where g=[R*BR/BPhi, R*BZ/BPhi].");

    m.def("progress_DX_pol_along_orbit", &py_progress_DX_pol_along_orbit,
        py::arg("R_traj"), py::arg("Z_traj"), py::arg("phi_traj"),
        py::arg("BR"), py::arg("BZ"), py::arg("BPhi"),
        py::arg("R_grid"), py::arg("Z_grid"), py::arg("Phi_grid"),
        py::arg("max_step") = 0.005,
        "Progress DX_pol(phi_e, phi_s) along an already sampled orbit.\n"
        "Returns ndarray of shape (N, 2, 2), with DX_pol[0] = identity.\n"
        "The orbit samples are used as the path; this does not retrace field lines.");

    m.def("progress_delta_X_along_orbit", &py_progress_delta_X_along_orbit,
        py::arg("R_traj"), py::arg("Z_traj"), py::arg("phi_traj"),
        py::arg("delta_X0"),
        py::arg("BR"), py::arg("BZ"), py::arg("BPhi"),
        py::arg("dBR"), py::arg("dBZ"), py::arg("dBPhi"),
        py::arg("R_grid"), py::arg("Z_grid"), py::arg("Phi_grid"),
        py::arg("max_step") = 0.005,
        "Progress delta_X along an already sampled orbit using\n"
        "d(delta_X)/dphi = d(RBpol/Bphi)/d(R,Z) delta_X + delta(RBpol/Bphi).\n"
        "dBR, dBZ and dBPhi are first-order delta-B component arrays.");

	    m.def("evolve_delta_X_cycle_along_cycle", &py_evolve_delta_X_cycle_along_cycle,
	        py::arg("R_traj"), py::arg("Z_traj"), py::arg("phi_traj"),
	        py::arg("delta_X_cyc0"),
	        py::arg("BR"), py::arg("BZ"), py::arg("BPhi"),
        py::arg("dBR"), py::arg("dBZ"), py::arg("dBPhi"),
        py::arg("R_grid"), py::arg("Z_grid"), py::arg("Phi_grid"),
        py::arg("max_step") = 0.005,
	        "Evolve an already closed periodic cycle displacement delta_X_cyc(phi).\n"
	        "This intentionally uses the same inhomogeneous ODE as\n"
	        "progress_delta_X_along_orbit; the difference is semantic: delta_X_cyc0\n"
	        "is the periodic initial displacement from the cycle closure solve.");

	    m.def("evolve_delta_X_cycle_along_orbit", &py_evolve_delta_X_cycle_along_cycle,
	        py::arg("R_traj"), py::arg("Z_traj"), py::arg("phi_traj"),
	        py::arg("delta_X_cyc0"),
	        py::arg("BR"), py::arg("BZ"), py::arg("BPhi"),
	        py::arg("dBR"), py::arg("dBZ"), py::arg("dBPhi"),
	        py::arg("R_grid"), py::arg("Z_grid"), py::arg("Phi_grid"),
	        py::arg("max_step") = 0.005,
	        "Compatibility alias for evolve_delta_X_cycle_along_cycle.");

    m.def("trace_surface_metrics_batch_twall", &py_trace_surface_metrics_batch_twall,
        py::arg("R_seeds"), py::arg("Z_seeds"),
        py::arg("R_axis"), py::arg("Z_axis"),
        py::arg("phi_start"), py::arg("N_turns"), py::arg("DPhi"),
        py::arg("BR"), py::arg("BZ"), py::arg("BPhi"),
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
        py::arg("BR"), py::arg("BZ"), py::arg("BPhi"),
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
           double phi_span, double dphi_out, int m_turns_DPm,
           double DPhi, double fd_eps,
           py::array_t<double> BR, py::array_t<double> BZ, py::array_t<double> BPhi,
           py::array_t<double> R_grid, py::array_t<double> Z_grid,
           py::array_t<double> Phi_grid) -> py::tuple
	        {
	            if (!std::isfinite(dphi_out) || std::abs(dphi_out) <= 1e-14)
	                throw std::runtime_error("dphi_out must be finite and non-zero");
	            if (!std::isfinite(DPhi) || std::abs(DPhi) <= 1e-14)
	                throw std::runtime_error("DPhi must be finite and non-zero");
	            const double dphi_out_eff = (phi_span < 0.0)
	                ? -std::abs(dphi_out)
	                :  std::abs(dphi_out);
	            int n_out = (int)std::ceil(std::abs(phi_span) / std::abs(dphi_out_eff)) + 1;
	            py::array_t<double> R_t({n_out}), Z_t({n_out}), phi_t({n_out});
	            py::array_t<double> DPm_t({n_out, 4});
	            py::array_t<int>    alive_t({n_out});

	            cyna::trace_orbit_along_phi(
	                R0, Z0, phi0, phi_span, dphi_out_eff, m_turns_DPm, DPhi, fd_eps,
                buf(BR,"BR"), buf(BZ,"BZ"), buf(BPhi,"BPhi"),
                buf(R_grid,"R_grid"), (int)R_grid.size(),
                buf(Z_grid,"Z_grid"), (int)Z_grid.size(),
                buf(Phi_grid,"Phi_grid"), (int)Phi_grid.size(),
                n_out,
                R_t.mutable_data(), Z_t.mutable_data(), phi_t.mutable_data(),
                DPm_t.mutable_data(), alive_t.mutable_data());

            return py::make_tuple(R_t, Z_t, phi_t, DPm_t, alive_t);
        },
        py::arg("R0"), py::arg("Z0"), py::arg("phi0"),
        py::arg("phi_span"), py::arg("dphi_out"), py::arg("m_turns_DPm"),
        py::arg("DPhi") = 0.05, py::arg("fd_eps") = 1e-4,
        py::arg("BR"), py::arg("BZ"), py::arg("BPhi"),
        py::arg("R_grid"), py::arg("Z_grid"), py::arg("Phi_grid"),
        "Integrate field line from (R0,Z0,phi0) for phi_span radians, outputting\n"
        "(R,Z,phi,DPm[n,4],alive[n]) at evenly-spaced phi_out intervals.\n"
        "DPm(φ)=DX_pol(φ,φ+2π·m_turns_DPm) via analytic DX_pol evolution — used for cycle visualisation.");

    m.def("trace_poincare_dpk_growth", &py_trace_poincare_dpk_growth,
        py::arg("R0"), py::arg("Z0"), py::arg("phi0"),
        py::arg("max_returns"),
        py::arg("return_period") = 2.0 * M_PI,
        py::arg("record_stride") = 1,
        py::arg("DPhi") = 0.05,
        py::arg("BR"), py::arg("BZ"), py::arg("BPhi"),
        py::arg("R_grid"), py::arg("Z_grid"), py::arg("Phi_grid"),
        "Trace one seed and record cumulative DP^k at Poincare returns.\n"
        "Returns (k, R, Z, phi, DPk[n,4], eig_abs[n,2], alive[n]).\n"
        "This integrates the orbit and variational equation once, so k=1..500\n"
        "is O(k) rather than repeated Python-level DP^m tracing.");

    m.def("trace_poincare_dpk_growth_twall", &py_trace_poincare_dpk_growth_twall,
        py::arg("R0"), py::arg("Z0"), py::arg("phi0"),
        py::arg("max_returns"),
        py::arg("return_period") = 2.0 * M_PI,
        py::arg("record_stride") = 1,
        py::arg("DPhi") = 0.05,
        py::arg("BR"), py::arg("BZ"), py::arg("BPhi"),
        py::arg("R_grid"), py::arg("Z_grid"), py::arg("Phi_grid"),
        py::arg("wall_phi_centers"), py::arg("wall_R"), py::arg("wall_Z"),
        py::arg("stop_at_wall") = true,
        "Wall-aware cumulative DP^k tracing.\n"
        "Returns (k, R, Z, phi, DPk[n,4], eig_abs[n,2], alive[n], hit[4], term[1]).\n"
        "hit=[R,Z,phi,k_float] records first wall crossing. term: 0=none, 1=wall, 2=grid/nonfinite.\n"
        "With stop_at_wall=False, tracing continues after the first wall hit.");

    m.def("compute_cycle_perturbation_shift", &py_compute_cycle_perturbation_shift,
        py::arg("R0"), py::arg("Z0"), py::arg("phi0"),
        py::arg("phi_span"), py::arg("dphi_out"),
        py::arg("DPhi") = 0.05, py::arg("fd_eps") = 1e-4,
        py::arg("BR_base"), py::arg("BZ_base"), py::arg("BPhi_base"),
        py::arg("BR_pert"), py::arg("BZ_pert"), py::arg("BPhi_pert"),
        py::arg("R_grid"), py::arg("Z_grid"), py::arg("Phi_grid"),
        "Integrate the first-order field-line response along one cycle.\n"
        "Returns (R, Z, phi, DP[n,4], dXpol[n,2], dXcyc[n,2], dXcyc0[2], alive[n]).\n"
        "dXpol is the zero-initial particular solution; dXcyc is the periodic\n"
        "cycle displacement satisfying (I-DP) dXcyc0 = dXpol(end).");

    // -----------------------------------------------------------------------
    // evolve_DPm_along_cycle — integrate DPm along a known orbit via commutator ODE
    // -----------------------------------------------------------------------
    m.def("evolve_DPm_along_cycle",
        [](py::array_t<double> R_traj, py::array_t<double> Z_traj,
           py::array_t<double> phi_traj,
           py::array_t<double> DPm_init,
           py::array_t<double> BR, py::array_t<double> BZ, py::array_t<double> BPhi,
           py::array_t<double> R_grid, py::array_t<double> Z_grid,
           py::array_t<double> Phi_grid)
        -> py::array_t<double>
        {
            int n_pts = (int)R_traj.size();
            py::array_t<double> DPm_out({n_pts, 4});
            cyna::evolve_DPm_along_cycle(
                buf(R_traj,"R_traj"), buf(Z_traj,"Z_traj"), buf(phi_traj,"phi_traj"),
                n_pts,
                buf(DPm_init,"DPm_init"),
                DPm_out.mutable_data(),
                buf(BR,"BR"), buf(BZ,"BZ"), buf(BPhi,"BPhi"),
                buf(R_grid,"R_grid"), (int)R_grid.size(),
                buf(Z_grid,"Z_grid"), (int)Z_grid.size(),
                buf(Phi_grid,"Phi_grid"), (int)Phi_grid.size());
            return DPm_out;
        },
        py::arg("R_traj"), py::arg("Z_traj"), py::arg("phi_traj"),
        py::arg("DPm_init"),
        py::arg("BR"), py::arg("BZ"), py::arg("BPhi"),
        py::arg("R_grid"), py::arg("Z_grid"), py::arg("Phi_grid"),
        "Integrate DPm along a known cycle orbit using the commutator ODE\n"
        "d(DPm)/dφ = J·DPm - DPm·J.  Requires only the orbit (R,Z,φ) and\n"
        "initial DPm from Newton at φ=0 — no additional field-line tracing.");

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

            py::array_t<float> BR({N}), BZ({N}), BPhi({N});
            float* pBR   = BR.mutable_data();
            float* pBZ   = BZ.mutable_data();
            float* pBPhi = BPhi.mutable_data();

            auto& cuda = _cuda_backend();
            bool ok = false;
            if (cuda.circular) {
                ok = cuda.circular(
                    (float)cx,(float)cy,(float)cz,
                    (float)nx,(float)ny,(float)nz,
                    (float)radius,(float)current,
                    pxyz, N, pBR, pBZ, pBPhi);
            }
            if (!ok) {
                cyna::circular_coil_field_cpu(
                    (float)cx,(float)cy,(float)cz,
                    (float)nx,(float)ny,(float)nz,
                    (float)radius,(float)current,
                    pxyz, N, pBR, pBZ, pBPhi);
            }
            return py::make_tuple(BR, BZ, BPhi);
        },
        py::arg("cx"),py::arg("cy"),py::arg("cz"),
        py::arg("nx"),py::arg("ny"),py::arg("nz"),
        py::arg("radius"),py::arg("current"),
        py::arg("xyz"),
        "Compute the magnetic field of one circular ring coil at N Cartesian\n"
        "field points. Uses a runtime CUDA backend when available, otherwise\n"
        "falls back to the OpenMP CPU implementation.\n"
        "Returns (BR, BZ, BPhi) — three (N,) float32 arrays in Tesla.");

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

            py::array_t<float> BR({N}), BZ({N}), BPhi({N});
            float* pBR   = BR.mutable_data();
            float* pBZ   = BZ.mutable_data();
            float* pBPhi = BPhi.mutable_data();

            auto& cuda = _cuda_backend();
            bool ok = false;
            if (cuda.biot) {
                ok = cuda.biot(
                    pss, pse, N_seg, (float)current,
                    pxyz, N, pBR, pBZ, pBPhi);
            }
            if (!ok) {
                cyna::biot_savart_field_cpu(pss, pse, N_seg, (float)current,
                                            pxyz, N, pBR, pBZ, pBPhi);
            }
            return py::make_tuple(BR, BZ, BPhi);
        },
        py::arg("seg_starts"),py::arg("seg_ends"),py::arg("current"),py::arg("xyz"),
        "Compute Biot-Savart field for one arbitrary filamentary coil.\n"
        "seg_starts/seg_ends: (N_seg,3) float32 Cartesian segment endpoints (m).\n"
        "Returns (BR, BZ, BPhi) — three (N,) float32 arrays in Tesla.\n"
        "Uses a runtime CUDA backend when available.");

    m.def("cuda_backend_available", []() { return _cuda_backend_available(); },
          "Return True when the optional runtime CUDA backend is loadable.");
}
