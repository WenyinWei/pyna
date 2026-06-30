@echo off
if not defined VCVARS64 (
  set "VCVARS64=%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
)
if exist "%VCVARS64%" call "%VCVARS64%"
if not defined PYTHON_INC set "PYTHON_INC=%LocalAppData%\Programs\Python\Python313\Include"
if not defined PYTHON_LIB set "PYTHON_LIB=%LocalAppData%\Programs\Python\Python313\libs"
if not defined PYBIND11_INC set "PYBIND11_INC=%LocalAppData%\Programs\Python\Python313\Lib\site-packages\pybind11\include"
set "REPO_ROOT=%~dp0"
cl /std:c++17 /O2 /openmp /EHsc /MD /LD ^
   /I"%PYTHON_INC%" /I"%PYBIND11_INC%" /I"%REPO_ROOT%cyna\include" ^
   "%REPO_ROOT%cyna\bindings\flt_bindings.cpp" ^
   /Fe:"%REPO_ROOT%pyna\_cyna\_cyna_ext.pyd" ^
   /link "%PYTHON_LIB%\python313.lib"
