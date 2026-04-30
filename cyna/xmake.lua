-- cyna xmake build
-- Compatible with xmake v2.x and v3.x.
-- Python paths and pybind11 headers are passed via environment variables
-- set by setup.py, avoiding any CMake/package-manager dependency.
set_project("cyna")
set_version("0.6.0")

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

-- pybind11 Python extension: bare _cyna_ext.pyd / _cyna_ext.so
-- Python's import system searches both bare names and ABI-tagged names,
-- so bare .pyd/.so works on all platforms and is simpler to produce.
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
        add_includedirs(PYBIND11_INC)
    else
        add_packages("pybind11")
    end

    -- Set the output filename to exactly what Python expects.
    -- We use bare .pyd/.so (without ABI tags) which Python's import system
    -- accepts as a fallback on all platforms. The wheel's ABI tag is encoded
    -- in the wheel filename itself (set by cibuildwheel), not the .so name.
    -- This is simpler and more portable than trying to replicate EXT_SUFFIX
    -- logic inside xmake Lua.
    set_basename("_cyna_ext")
    if is_plat("windows") then
        set_extension(".pyd")
    else
        set_prefixname("")   -- suppress "lib" prefix on Linux
        set_extension(".so")
    end
    -- Windows-specific linker flags (applied regardless of extension source)
    if is_plat("windows") then
        -- Console subsystem is required; GUI causes "DLL load failed"
        add_ldflags("/SUBSYSTEM:CONSOLE", {force = true, tools = {"link"}})
        -- Suppress .exp and .lib side outputs
        add_ldflags("/NOEXP", {force = true, tools = {"link"}})
    elseif is_plat("macosx") then
        add_ldflags("-bundle", "-undefined dynamic_lookup", {force = true})
    end

    -- Compiler flags
    set_languages("c++17")
    add_cxxflags("/O2", "/openmp",  {tools = {"cl"}})
    add_cxxflags("-O3", "-fopenmp", {tools = {"gcc"}})
    add_cxxflags("-O3",             {tools = {"clang"}})

    -- Link Python runtime
    if is_plat("windows") then
        if PY_LIB_WIN ~= "" then
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

    -- CUDA (optional): compile coil_field_cuda.cu and link cudart
    if has_config("with-cuda") then
        add_defines("CYNA_CUDA_ENABLED")
        add_rules("cuda")
        add_files("coil_field_cuda.cu")
        -- sm_86 = Ampere (RTX 3060/3070/3080/A40 …); adjust for other GPUs
        add_cuflags("-arch=sm_86", "--use_fast_math", "-allow-unsupported-compiler", {force = true})
        if is_plat("windows") then
            add_links("cudart_static", "cuda")
        else
            add_links("cudart")
        end
    end

    -- After build: copy _cyna_ext.pyd/.so into pyna/_cyna/
    after_build(function(target)
        local dest = path.join(os.scriptdir(), "..", "pyna", "_cyna")
        os.mkdir(dest)
        local src = target:targetfile()
        os.cp(src, dest)
        print("cyna: installed " .. path.filename(src) .. " -> " .. dest)
    end)
