# Cyna — C++ Field Line Tracer

This directory contains the **Cyna** C++ module, migrated from [MHDcxx](https://github.com/WenyinWei/MHDcxx).

Cyna provides:
- **`include/cyna/flt.hpp`** — C++ field line tracer (FLT) with Ascent ODE engine + OpenMP/TBB parallelism
- **`include/cyna/interpolate.hpp`** — `RegularGridInterpolator` in C++ (xtensor)
- **`include/cyna/io.hpp`** — npz I/O in C++
- **`include/BS_thread_pool.hpp`** — BS thread pool header
- **`app/flt3d.cpp`** — 3D FLT application
- **`app/construct_flux_coordinate.cpp`** — Flux coordinate construction

## Build

Uses [xmake](https://xmake.io/) or CMake. Requires:
- [xtensor](https://github.com/xtensorstack/xtensor)
- [cnpy](https://github.com/rogersce/cnpy)
- [Ascent ODE engine](https://github.com/WenyinWei/Ascent)
- OpenMP or TBB (for parallelism)
