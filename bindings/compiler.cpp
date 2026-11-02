#include "compiler_bridge.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;
void bindRuntime(nb::module_ &module);

NB_MODULE(_native, module) {
  module.doc() = "ASK native compiler and persistent device runtime";
  module.def("capabilities", &compilerCapabilitiesJSON,
             nb::call_guard<nb::gil_scoped_release>());
  module.def("prepare", &prepareJSON, nb::arg("source"), nb::arg("optimize") = false,
     nb::arg("fold_constants") = true, nb::arg("eliminate_dead_code") = true,
     nb::arg("fuse_elementwise") = true, nb::call_guard<nb::gil_scoped_release>());
  module.def("partition", &partitionJSON, nb::arg("source"), nb::arg("optimize") = true,
     nb::arg("fold_constants") = true, nb::arg("eliminate_dead_code") = true,
     nb::arg("fuse_elementwise") = true,
     nb::call_guard<nb::gil_scoped_release>());
  bindRuntime(module);
}
