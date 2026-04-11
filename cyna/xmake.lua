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
-- EXT_SUFFIX: the correct Python extension suffix, e.g.
--   Windows:  .pyd  (or .cp310-win_amd64.pyd for limited API builds)
--   Linux:    .cpython-310-x86_64-linux-gnu.so
--   macOS:    .cpython-310-darwin.so
-- setup.py exports this so xmake can name the output file correctly.
local EXT_SUFFIX = os.getenv("CYNA_EXT_SUFFIX") or ""

if PY_INC ~= "" then
    print("cyna: Python include  : " .. PY_INC)
end
if EXT_SUFFIX ~= "" then
    print("cyna: EXT_SUFFIX      : " .. EXT_SUFFIX)
end
if is_plat("windows") and PY_LIB_WIN ~= "" then
    print("cyna: Python lib (win): " .. PY_LIB_WIN)
end

-- Header-only cyna library
target("cyna")
    set_kind("headeronly")
    add_includedirs("include", {public = true})
    add_headerfiles("include/(cyna/*.hpp)")

-- pybind11 Python extension
-- CRITICAL: always use add_rules("python.module") to ensure:
--   1. Correct PE subsystem (console, not GUI) on Windows
--   2. Correct .pyd / .so / .cpython-XY-*.so suffix per platform
--   3. Correct linker flags (no default lib conflicts on Windows)
-- We use set_basename("_cyna_ext") only; let the rule set the suffix.
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
    -- EXT_SUFFIX (from CYNA_EXT_SUFFIX env var) includes the leading dot,
    -- e.g. ".pyd" or ".cp310-win_amd64.pyd" or ".cpython-310-...so"
    set_basename("_cyna_ext")
    if EXT_SUFFIX ~= "" then
        -- Use the exact suffix Python reported for this interpreter
        set_extension(EXT_SUFFIX)
    elseif is_plat("windows") then
        set_extension(".pyd")
    elseif is_plat("macosx") then
        set_extension(".so")
    else
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

    -- CUDA (optional)
    if has_config("with-cuda") then
        add_defines("CYNA_CUDA_ENABLED")
    end

    -- After build: copy the .pyd/.so into pyna/_cyna/
    -- python.module produces e.g. _cyna_ext.pyd (Windows) or
    -- _cyna_ext.cpython-310-x86_64-linux-gnu.so (Linux).
    -- We copy whatever file was produced; Python's import system finds it.
    after_build(function(target)
        local dest = path.join(os.scriptdir(), "..", "pyna", "_cyna")
        os.mkdir(dest)
        local src = target:targetfile()
        os.cp(src, dest)
        print("cyna: installed " .. path.filename(src) .. " -> " .. dest)
    end)
