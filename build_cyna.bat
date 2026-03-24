call "D:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
cl /std:c++17 /O2 /openmp /EHsc /MD /LD ^
   /I"C:\Users\Legion\AppData\Local\Programs\Python\Python313\Include" /I"C:\Users\Legion\AppData\Local\Programs\Python\Python313\Lib\site-packages\pybind11\include" /I"D:\Repo\pyna\cyna\include" ^
   "D:\Repo\pyna\cyna\bindings\flt_bindings.cpp" ^
   /Fe:"D:\Repo\pyna\pyna\_cyna\_cyna_ext.pyd" ^
   /link "C:\Users\Legion\AppData\Local\Programs\Python\Python313\libs\python313.lib"
