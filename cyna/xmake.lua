-- cyna xmake build
set_project("cyna")
set_version("0.1.0")

-- C++17 required
set_languages("c++17")

-- Dependencies (optional: xtensor, openmp)
add_requires("xtensor", {optional = true})
add_requires("openmp", {optional = true})

-- Header-only library target
target("cyna")
    set_kind("headeronly")
    add_includedirs("include", {public = true})
    add_headerfiles("include/(cyna/*.hpp)")

-- FLT standalone app
target("flt3d")
    set_kind("binary")
    add_files("app/flt3d.cpp")
    add_deps("cyna")
    add_packages("xtensor")

-- construct_flux_coordinate standalone app
target("construct_flux_coordinate")
    set_kind("binary")
    add_files("app/construct_flux_coordinate.cpp")
    add_deps("cyna")
    add_packages("xtensor")

-- pybind11 bindings (optional)
target("cyna_python")
    set_kind("shared")
    add_files("bindings/flt_bindings.cpp")
    add_deps("cyna")
    -- set_filename("_cyna_ext")
    -- add_packages("pybind11")
    add_rules("python.library")
