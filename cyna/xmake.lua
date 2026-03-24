-- cyna xmake build
set_project("cyna")
set_version("0.2.0")

set_languages("c++17")

-- pybind11 for Python bindings
add_requires("pybind11")

-- Header-only library target
target("cyna")
    set_kind("headeronly")
    add_includedirs("include", {public = true})
    add_headerfiles("include/(cyna/*.hpp)")

-- pybind11 bindings
target("cyna_python")
    set_kind("shared")
    add_files("bindings/flt_bindings.cpp")
    add_deps("cyna")
    add_packages("pybind11")
    add_includedirs("include")
    set_filename("_cyna_ext")
    add_rules("python.library")
    set_languages("c++17")
    add_cxxflags("/O2", {tools = {"cl"}})
    add_cxxflags("-O3", {tools = {"gcc", "clang"}})
