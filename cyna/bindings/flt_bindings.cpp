// cyna Python bindings via pybind11
// Build: cd cyna && xmake build cyna_python
// Then: copy _cyna_ext.*.so (or .pyd on Windows) to pyna/_cyna/
//
// Currently a stub — full implementation pending pybind11 integration.
// The Python fallback in pyna/_cyna/__init__.py handles all cases.

#ifdef PYBIND11_VERSION_MAJOR  // only compiled when pybind11 is available

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cyna/flt.hpp>

namespace py = pybind11;

PYBIND11_MODULE(_cyna_ext, m) {
    m.doc() = "cyna: C++ acceleration layer for pyna field-line tracing";

    // TODO: expose trace_fieldlines_batch() when ready
    // m.def("trace_fieldlines_batch", &cyna::flt::trace_fieldlines_batch, ...);

    m.attr("__version__") = "0.1.0";
    m.attr("available") = true;
}

#endif
