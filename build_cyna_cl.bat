@echo off
if not defined VCVARS64 (
  set "VCVARS64=%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
)
if exist "%VCVARS64%" call "%VCVARS64%"

if not defined PYBIND11_INC set "PYBIND11_INC=%LocalAppData%\Programs\Python\Python313\Lib\site-packages\pybind11\include"
if not defined PYTHON_INC set "PYTHON_INC=%LocalAppData%\Programs\Python\Python313\Include"
if not defined PYTHON_LIB set "PYTHON_LIB=%LocalAppData%\Programs\Python\Python313\libs"
set "CYNA_INC=%~dp0cyna\include"

cd /d "%~dp0cyna"

cl /O2 /std:c++17 /EHsc /LD ^
  /I"%PYBIND11_INC%" ^
  /I"%PYTHON_INC%" ^
  /I"%CYNA_INC%" ^
  bindings/flt_bindings.cpp ^
  /link /LIBPATH:"%PYTHON_LIB%" ^
  /OUT:_cyna_ext.pyd ^
  /DLL

echo Build exit code: %ERRORLEVEL%
