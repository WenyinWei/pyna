// flt_bindings.cpp — pybind11 bindings for cyna Poincaré tracer
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <thread>
#include <stdexcept>
#include "cyna/poincare.hpp"

namespace py = pybind11;

// Helper: assert contiguous double array
static const double* buf(const py::array_t<double>& a, const char* name) {
    if (!(a.flags() & py::array::c_style))
        throw std::runtime_error(std::string(name) + " must be C-contiguous");
    return a.data();
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
                R_hb.mutable_data(), Z_hb.mutable_data(), phi_hb.mutable_data());

            return py::make_tuple(L_fwd, L_bwd,
                                  R_hf, Z_hf, phi_hf,
                                  R_hb, Z_hb, phi_hb);
        },
        py::arg("R_seeds"), py::arg("Z_seeds"), py::arg("phi_start"),
        py::arg("max_turns"), py::arg("DPhi"),
        py::arg("BR"), py::arg("BPhi"), py::arg("BZ"),
        py::arg("R_grid"), py::arg("Z_grid"), py::arg("Phi_grid"),
        py::arg("wall_phi_centers"), py::arg("wall_R"), py::arg("wall_Z"),
        py::arg("n_threads") = -1,
        "Like trace_connection_length_twall but also returns wall-hit coordinates.\n"
        "Returns (L_fwd, L_bwd, R_hit_fwd, Z_hit_fwd, phi_hit_fwd,\n"
        "                       R_hit_bwd, Z_hit_bwd, phi_hit_bwd).\n"
        "NaN where field line does not terminate within max_turns.");
}
