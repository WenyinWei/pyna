"""pyna setup.py — auto-builds the cyna C++ extension via xmake.

Strategy
--------
1. Ensure xmake is installed (platform-specific bootstrap if missing).
2. Ensure a C++ compiler is available (platform-specific install if missing).
3. Run ``xmake build`` inside the ``cyna/`` subdirectory.
4. Copy the built ``_cyna_ext.{pyd,so}`` into ``pyna/_cyna/``.

The build is entirely opt-in with graceful degradation: if anything fails,
pyna still installs but cyna (the C++ accelerator) is simply unavailable,
and ``pyna._cyna.is_available()`` returns False.  All hot paths in pyna fall
back to pure-Python implementations in that case.

CUDA support is auto-detected: if ``nvcc`` is on PATH the CUDA path is
compiled in automatically; no user action required.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

from setuptools import setup, Extension

# Declare a stub Extension so cibuildwheel / pip recognise pyna as a
# binary package and trigger our custom build_ext command.
# The actual compilation is done entirely inside BuildCynaExt.run() via
# xmake; setuptools never sees (and never compiles) this stub.
_CYNA_STUB = Extension(
    name="pyna._cyna._cyna_ext",
    sources=[],          # xmake handles sources; setuptools touches nothing
    optional=True,       # build failure is non-fatal
)
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.develop import develop as _develop
from setuptools.command.install import install as _install

# ── Repository layout ────────────────────────────────────────────────────────

ROOT      = Path(__file__).parent.resolve()
CYNA_DIR  = ROOT / "cyna"
DEST_DIR  = ROOT / "pyna" / "_cyna"
XMAKE_LUA = CYNA_DIR / "xmake.lua"


# ═══════════════════════════════════════════════════════════════════════════════
# Platform helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _run(cmd: list[str], **kwargs) -> int:
    """Run a subprocess, stream output, return exit code."""
    print(f"[cyna-build] $ {' '.join(str(c) for c in cmd)}", flush=True)
    result = subprocess.run(cmd, **kwargs)
    return result.returncode


def _is_windows() -> bool:
    return platform.system() == "Windows"


def _is_macos() -> bool:
    return platform.system() == "Darwin"


def _is_linux() -> bool:
    return platform.system() == "Linux"


# ── xmake ─────────────────────────────────────────────────────────────────────

def _xmake_path() -> str | None:
    """Return path to xmake binary, or None if not found."""
    # Common install locations beyond PATH
    extra = []
    if _is_windows():
        extra += [
            Path(os.environ.get("USERPROFILE", "")) / ".xmake" / "bin" / "xmake.exe",
            Path("C:/Users") / os.environ.get("USERNAME", "") / ".xmake" / "bin" / "xmake.exe",
        ]
    else:
        extra += [
            Path.home() / ".local" / "bin" / "xmake",
            Path("/usr/local/bin/xmake"),
        ]
    for p in extra:
        if Path(p).is_file():
            return str(p)
    return shutil.which("xmake")


def _install_xmake() -> bool:
    """Bootstrap xmake via its official one-liner installer.

    Returns True on success.
    """
    print("[cyna-build] xmake not found — installing …", flush=True)

    if _is_windows():
        # winget (Windows 11+) -> Chocolatey -> PowerShell installer (Gitee CN mirror first)
        for cmd in [
            ["winget", "install", "--id", "xmake-io.xmake", "-e", "--silent",
             "--accept-source-agreements", "--accept-package-agreements"],
            ["choco", "install", "xmake", "-y"],
            # Gitee mirror (fast in CN)
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
             "-Command",
             "Invoke-Expression (Invoke-WebRequest -Uri 'https://gitee.com/tboox/xmake/raw/master/scripts/get.ps1' -UseBasicParsing).Content"],
            # GitHub fallback
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
             "-Command",
             "Invoke-Expression (Invoke-WebRequest -Uri 'https://xmake.io/shget.text' -UseBasicParsing).Content"],
        ]:
            if shutil.which(cmd[0]):
                rc = _run(cmd)
                if rc == 0 and _xmake_path():
                    return True

    elif _is_macos():
        for cmd in [
            ["brew", "install", "xmake"],
            # Gitee mirror (CN), then GitHub
            ["bash", "-c", "curl -fsSL https://gitee.com/tboox/xmake/raw/master/scripts/get.sh | bash"],
            ["bash", "-c", "curl -fsSL https://xmake.io/shget.text | bash"],
        ]:
            if shutil.which(cmd[0]) or cmd[0] == "bash":
                rc = _run(cmd, shell=False)
                if rc == 0 and _xmake_path():
                    return True

    else:  # Linux
        for cmd in [
            # Distro package managers
            ["apt-get", "install", "-y", "xmake"],
            ["dnf",     "install", "-y", "xmake"],
            ["pacman",  "-S",      "--noconfirm", "xmake"],
            # Gitee mirror (CN), then GitHub
            ["bash", "-c", "curl -fsSL https://gitee.com/tboox/xmake/raw/master/scripts/get.sh | bash"],
            ["bash", "-c", "curl -fsSL https://xmake.io/shget.text | bash"],
        ]:
            mgr = cmd[0]
            if mgr == "bash" or shutil.which(mgr):
                rc = _run(cmd, shell=False)
                if rc == 0 and _xmake_path():
                    return True

    # pip-installable wrapper as last resort
    rc = _run([sys.executable, "-m", "pip", "install", "xmake-cli", "--quiet"])
    if rc == 0 and _xmake_path():
        return True

    print("[cyna-build] WARNING: could not install xmake automatically.", flush=True)
    return False


# ── C++ compiler ──────────────────────────────────────────────────────────────

def _has_cxx_compiler() -> bool:
    """Return True if a usable C++ compiler is on PATH or in a standard VS install."""
    if _is_windows():
        # Check PATH first
        if any(shutil.which(c) for c in ["cl", "clang-cl", "g++", "c++"]):
            return True
        # Search standard VS install locations (VS 2019/2022/2026)
        import glob
        for pattern in [
            r"C:\Program Files\Microsoft Visual Studio\*\*\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe",
            r"C:\Program Files (x86)\Microsoft Visual Studio\*\*\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe",
            r"C:\BuildTools\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe",
        ]:
            if glob.glob(pattern):
                return True
        return False
    else:
        return any(shutil.which(c) for c in ["g++", "clang++", "c++"])


def _install_cxx_compiler() -> bool:
    """Install a minimal C++ compiler for the current platform.

    Returns True on success.
    """
    print("[cyna-build] No C++ compiler found — installing …", flush=True)

    if _is_windows():
        # VS Build Tools 2022 via winget (silent, C++ workload only)
        # Fallback: Chocolatey visualstudio2022buildtools
        for cmd in [
            ["winget", "install", "--id",
             "Microsoft.VisualStudio.2022.BuildTools",
             "--override",
             "--quiet --add Microsoft.VisualStudio.Workload.VCTools"
             " --includeRecommended --wait",
             "--silent", "--accept-source-agreements",
             "--accept-package-agreements"],
            ["choco", "install", "visualstudio2022buildtools",
             "--package-parameters",
             "--add Microsoft.VisualStudio.Workload.VCTools"
             " --includeRecommended",
             "-y"],
        ]:
            if shutil.which(cmd[0]):
                rc = _run(cmd)
                if rc == 0 and _has_cxx_compiler():
                    return True
        # MinGW-w64 as a lighter fallback
        for cmd in [
            ["winget", "install", "--id", "GnuWin32.Make", "-e",
             "--silent", "--accept-source-agreements", "--accept-package-agreements"],
            ["choco", "install", "mingw", "-y"],
        ]:
            if shutil.which(cmd[0]):
                rc = _run(cmd)
                if rc == 0 and _has_cxx_compiler():
                    return True

    elif _is_macos():
        # Xcode command-line tools
        rc = _run(["xcode-select", "--install"])
        if _has_cxx_compiler():
            return True
        # Homebrew llvm
        if shutil.which("brew"):
            rc = _run(["brew", "install", "llvm"])
            if _has_cxx_compiler():
                return True

    else:  # Linux
        for cmd in [
            ["apt-get", "install", "-y", "build-essential"],
            ["dnf",     "install", "-y", "gcc-c++", "make"],
            ["pacman",  "-S",      "--noconfirm", "base-devel"],
            ["apk",     "add",     "build-base"],
        ]:
            mgr = cmd[0]
            if shutil.which(mgr):
                rc = _run(cmd)
                if rc == 0 and _has_cxx_compiler():
                    return True

    print("[cyna-build] WARNING: could not install C++ compiler automatically.", flush=True)
    return False


# ── CUDA (optional) ───────────────────────────────────────────────────────────

def _has_cuda() -> bool:
    return shutil.which("nvcc") is not None


# ── pybind11 (Python-side, for include path) ──────────────────────────────────

def _ensure_pybind11(skip_install: bool = False) -> None:
    """Make sure pybind11 is importable so xmake can find its include path.

    Uses CN-friendly mirrors when the standard index is unreachable.
    Pass skip_install=True (CI mode) to skip the install attempt.
    """
    try:
        import pybind11  # noqa: F401
        return  # already installed
    except ImportError:
        pass
    if skip_install:
        print("[cyna-build] WARNING: pybind11 not found and skip_install=True.", flush=True)
        return
    print("[cyna-build] Installing pybind11 ...", flush=True)
    # Try primary index first; fall back to Tsinghua mirror on failure
    for index_url in [
        None,  # default (PyPI)
        "https://pypi.tuna.tsinghua.edu.cn/simple",   # Tsinghua (CN)
        "https://mirrors.aliyun.com/pypi/simple",      # Aliyun (CN)
        "https://pypi.mirrors.ustc.edu.cn/simple",     # USTC (CN)
    ]:
        cmd = [sys.executable, "-m", "pip", "install", "pybind11", "--quiet"]
        if index_url:
            cmd += ["-i", index_url, "--trusted-host", index_url.split("/")[2]]
        rc = _run(cmd)
        if rc == 0:
            try:
                import pybind11  # noqa: F401
                return
            except ImportError:
                pass
    print("[cyna-build] WARNING: pybind11 install failed; build may fail.", flush=True)


# ── xmake build ───────────────────────────────────────────────────────────────

def _build_cyna() -> bool:
    """Run xmake to build _cyna_ext and copy the result into pyna/_cyna/.

    Returns True on success.
    """
    # In CI (cibuildwheel) xmake and the compiler are pre-installed;
    # skip the auto-install logic to save time and avoid network errors.
    _skip_tool_install = os.environ.get("CYNA_SKIP_TOOL_INSTALL", "") == "1"

    xmake = _xmake_path()
    if xmake is None:
        if _skip_tool_install:
            print("[cyna-build] ERROR: xmake not found and CYNA_SKIP_TOOL_INSTALL=1.", flush=True)
            return False
        if not _install_xmake():
            return False
        xmake = _xmake_path()
    if xmake is None:
        print("[cyna-build] ERROR: xmake not available after install attempt.", flush=True)
        return False

    if not _has_cxx_compiler():
        if _skip_tool_install:
            print("[cyna-build] WARNING: no C++ compiler found; proceeding (CI should have one).", flush=True)
        elif not _install_cxx_compiler():
            print("[cyna-build] WARNING: proceeding without C++ compiler — may fail.", flush=True)

    _ensure_pybind11(skip_install=_skip_tool_install)

    # Determine Python info for xmake
    py_inc  = sysconfig.get_path("include")
    py_lib  = sysconfig.get_config_var("LIBDIR") or ""
    py_ver  = f"{sys.version_info.major}{sys.version_info.minor}"
    ext_sfx = sysconfig.get_config_var("EXT_SUFFIX") or ".so"

    env = os.environ.copy()
    # Export Python paths as env vars so xmake.lua can read them
    # (xmake Lua sandbox forbids pcall/io.popen, so we pass via env)
    import sysconfig as _sc
    env["XMAKE_PYTHON"]       = sys.executable
    env["CYNA_PY_INC"]         = _sc.get_path("include") or ""
    env["CYNA_PY_LIBDIR"]      = _sc.get_config_var("LIBDIR") or ""
    # pybind11 headers from pip (avoids xmake's CMake-based package download)
    try:
        import pybind11 as _pb11
        env["CYNA_PYBIND11_INC"] = _pb11.get_include()
    except ImportError:
        env["CYNA_PYBIND11_INC"] = ""
    # Windows: find pythonXY.lib
    if _is_windows():
        import glob as _glob
        libs_dir = Path(sys.exec_prefix) / "libs"
        py_libs = list(libs_dir.glob("python*.lib"))
        env["CYNA_PY_LIB_WIN"] = str(py_libs[0]) if py_libs else ""
    else:
        env["CYNA_PY_LIB_WIN"] = ""

    # CUDA option: xmake custom option uses --with-cuda=y / n
    cuda_flag = ["--with-cuda=y"] if _has_cuda() else ["--with-cuda=n"]

    print(f"[cyna-build] Building cyna in {CYNA_DIR} …", flush=True)
    print(f"[cyna-build]   Python include : {py_inc}", flush=True)
    print(f"[cyna-build]   CUDA           : {'yes' if _has_cuda() else 'no'}", flush=True)

    # --require=no: skip all package-manager network access (pybind11 comes from pip)
    base_cfg = [xmake, "config", "--yes", "--mode=release", "--require=no"]
    if not _is_windows():
        base_cfg += [f"--python={sys.executable}"]
    rc = _run(base_cfg + cuda_flag, cwd=str(CYNA_DIR), env=env)
    if rc != 0:
        if cuda_flag:
            print("[cyna-build] Config with CUDA failed, retrying without ...", flush=True)
            rc = _run(base_cfg, cwd=str(CYNA_DIR), env=env)

    # Build
    rc = _run([xmake, "build", "cyna_python"], cwd=str(CYNA_DIR), env=env)
    if rc != 0:
        print("[cyna-build] xmake build failed.", flush=True)
        return False

    # The xmake after_build hook copies _cyna_ext.{pyd,so} into pyna/_cyna/.
    # Verify the file landed there.
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    dest_matches = (
        list(DEST_DIR.glob("_cyna_ext*.pyd")) +
        list(DEST_DIR.glob("_cyna_ext*.so")) +
        list(DEST_DIR.glob("_cyna_ext.pyd")) +
        list(DEST_DIR.glob("_cyna_ext.so"))
    )
    if dest_matches:
        print(f"[cyna-build] Extension installed: {dest_matches[0].name} in {DEST_DIR}",
              flush=True)
        return True

    # after_build hook already copied it but under a different name pattern
    # (xmake v3 uses cpython ABI suffix).  Accept any _cyna_ext* file.
    any_match = list(DEST_DIR.glob("_cyna_ext*"))
    non_debug = [f for f in any_match if f.suffix not in (".lib", ".exp", ".pdb")]
    if non_debug:
        print(f"[cyna-build] Extension installed: {non_debug[0].name}", flush=True)
        return True

    print("[cyna-build] WARNING: xmake built OK but extension not found in pyna/_cyna/ "
          "-- check xmake after_build output above.", flush=True)
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# Setuptools command hooks
# ═══════════════════════════════════════════════════════════════════════════════

class BuildCynaExt(_build_ext):
    """Custom build_ext that runs xmake to build _cyna_ext."""

    def run(self):
        try:
            ok = _build_cyna()
            if not ok:
                print(
                    "[cyna-build] cyna C++ extension was not built.  "
                    "pyna will work in pure-Python fallback mode (slower).",
                    flush=True,
                )
        except Exception as exc:
            print(
                f"[cyna-build] Exception during cyna build: {exc}\n"
                "  pyna will work in pure-Python fallback mode.",
                flush=True,
            )
        # Always call super so other extensions (if any) are handled normally
        # Don't fail the whole install if cyna fails
        try:
            super().run()
        except Exception:
            pass


class DevelopWithCyna(_develop):
    def run(self):
        try:
            _build_cyna()
        except Exception as exc:
            print(f"[cyna-build] build skipped in develop mode: {exc}", flush=True)
        super().run()


class InstallWithCyna(_install):
    def run(self):
        try:
            _build_cyna()
        except Exception as exc:
            print(f"[cyna-build] build skipped in install mode: {exc}", flush=True)
        super().run()


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

setup(
    ext_modules=[_CYNA_STUB],
    cmdclass={
        "build_ext": BuildCynaExt,
        "develop":   DevelopWithCyna,
        "install":   InstallWithCyna,
    },
)
