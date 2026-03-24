@echo off
call "D:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

set PYBIND11_INC=C:\Users\Legion\AppData\Local\Programs\Python\Python313\Lib\site-packages\pybind11\include
set PYTHON_INC=C:\Users\Legion\AppData\Local\Programs\Python\Python313\Include
set PYTHON_LIB=C:\Users\Legion\AppData\Local\Programs\Python\Python313\libs
set CYNA_INC=D:\Repo\pyna\cyna\include

cd /d D:\Repo\pyna\cyna

cl /O2 /std:c++17 /EHsc /LD ^
  /I"%PYBIND11_INC%" ^
  /I"%PYTHON_INC%" ^
  /I"%CYNA_INC%" ^
  bindings/flt_bindings.cpp ^
  /link /LIBPATH:"%PYTHON_LIB%" ^
  /OUT:_cyna_ext.pyd ^
  /DLL

echo Build exit code: %ERRORLEVEL%
