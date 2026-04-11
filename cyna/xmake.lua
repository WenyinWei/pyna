-- cyna xmake build
-- Compatible with xmake v2.x and v3.x.
-- Python paths and pybind11 headers are passed via environment variables
-- set by setup.py, avoiding any CMake/package-manager dependency.
set_project("cyna")
set_version("0.2.0")

set_languages("c++17")

-- Optional CUDA (off by default; setup.py enables when nvcc is present)
option("with-cuda")
    set_default(false)
    set_showmenu(true)
    set_description("Enable CUDA acceleration")
option_end()

-- pybind11: prefer pip-installed headers (CYNA_PYBIND11_INC env var).
-- Only fall back to xmake package repo (needs CMake) when unset.
local PYBIND11_INC = os.getenv("CYNA_PYBIND11_INC") or ""
if PYBIND11_INC == "" then
    add_requires("pybind11")
    print("cyna: pybind11 via xmake package repo")
else
    print("cyna: pybind11 include : " .. PYBIND11_INC)
end

-- Python paths (set by setup.py via env vars)
local PY_INC     = os.getenv("CYNA_PY_INC")     or ""
local PY_LIBDIR  = os.getenv("CYNA_PY_LIBDIR")  or ""
local PY_LIB_WIN = os.getenv("CYNA_PY_LIB_WIN") or ""

if PY_INC ~= "" then
    print("cyna: Python include  : " .. PY_INC)
end
if is_plat("windows") and PY_LIB_WIN ~= "" then
    print("cyna: Python lib (win): " .. PY_LIB_WIN)
end

-- Header-only cyna library
target("cyna")
    set_kind("headeronly")
    add_includedirs("include", {public = true})
    add_headerfiles("include/(cyna/*.hpp)")

-- pybind11 Python extension (_cyna_ext.pyd / _cyna_ext.so)
target("cyna_python")
    set_kind("shared")
    add_files("bindings/flt_bindings.cpp")
    add_deps("cyna")

    -- Include paths: cyna headers + Python + pybind11
    add_includedirs("include")
    if PY_INC ~= "" then
        add_includedirs(PY_INC)
    end
    if PYBIND11_INC ~= "" then
        -- Use pip pybind11 headers directly
        add_includedirs(PYBIND11_INC)
    else
        -- Fall back to xmake-managed pybind11 package
        add_packages("pybind11")
    end

    -- python.module rule (xmake v3) / python.library (v2) sets correct ext suffix
    set_filename("_cyna_ext")
    add_rules("python.module")

    -- Compiler flags
    set_languages("c++17")
    add_cxxflags("/O2", "/openmp",  {tools = {"cl"}})
    add_cxxflags("-O3", "-fopenmp", {tools = {"gcc"}})
    add_cxxflags("-O3",             {tools = {"clang"}})

    -- Link Python runtime
    if is_plat("windows") then
        if PY_LIB_WIN ~= "" then
            -- path.basename returns a string in xmake; tostring() for safety
            local libdir  = path.directory(PY_LIB_WIN)
            local libfile = tostring(path.basename(PY_LIB_WIN))
            local libname = libfile:gsub("%.lib$", "")
            add_linkdirs(libdir)
            add_links(libname)
        end
    elseif PY_LIBDIR ~= "" then
        add_linkdirs(PY_LIBDIR)
        add_rpathdirs(PY_LIBDIR)
    end

    -- CUDA (optional, enabled via --with-cuda=y)
    if has_config("with-cuda") then
        add_defines("CYNA_CUDA_ENABLED")
    end

    -- After build: copy into pyna/_cyna/ so Python can import it
    after_build(function(target)
        local dest = path.join(os.scriptdir(), "..", "pyna", "_cyna")
        os.mkdir(dest)
        local src = target:targetfile()
        os.cp(src, dest)
        print("cyna: installed " .. path.filename(src) .. " -> " .. dest)
    end)
