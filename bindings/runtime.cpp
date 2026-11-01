#include "ask/Runtime/CUDA.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace ask::runtime;

void bindRuntime(nb::module_ &module) {
  nb::exception<CUDAError>(module, "CUDAError", nb::module_::import_("ask.cuda").attr("CUDAError"));
  nb::class_<DeviceBuffer>(module,"CUDADeviceBuffer")
      .def_prop_ro("data_ptr",&DeviceBuffer::pointer)
      .def_prop_ro("nbytes",&DeviceBuffer::size)
      .def_prop_ro("readonly",&DeviceBuffer::readonly)
      .def("copy_from_host",[](DeviceBuffer &buffer,nb::bytes data){
        const void *pointer=data.data();std::size_t size=data.size();
        nb::gil_scoped_release release;buffer.copyFromHost(pointer,size);
      })
      .def("to_host_bytes",[](const DeviceBuffer &buffer){
        std::string data;{nb::gil_scoped_release release;data=buffer.copyToHost();}
        return nb::bytes(data.data(),data.size());
      });
  nb::class_<Kernel>(module,"CUDAKernel")
      .def_prop_ro("entry",&Kernel::entry)
      .def_prop_ro("max_threads",&Kernel::maxThreads);
  nb::class_<Event>(module,"CUDAEvent")
      .def("query",&Event::query,nb::call_guard<nb::gil_scoped_release>())
      .def("wait",&Event::wait,nb::call_guard<nb::gil_scoped_release>());
  nb::class_<Session>(module,"CUDASession")
      .def_static("create",&Session::create,nb::arg("device")=0,nb::call_guard<nb::gil_scoped_release>())
      .def("allocate",&Session::allocate,nb::arg("nbytes"),nb::call_guard<nb::gil_scoped_release>())
      .def("import_buffer",&Session::importBuffer,nb::arg("pointer"),nb::arg("nbytes"),
           nb::arg("readonly")=false,nb::call_guard<nb::gil_scoped_release>())
      .def("load",[](Session &session,nb::bytes data,const std::string &entry,
                      const std::vector<std::string> &types,int arch,unsigned parameters){
        std::string binary(static_cast<const char *>(data.data()),data.size());
        nb::gil_scoped_release release;return session.load(binary,entry,types,arch,parameters);
      },nb::arg("cubin"),nb::arg("entry"),nb::arg("required_types"),nb::arg("arch"),
        nb::arg("expected_parameters")=4)
      .def("submit",&Session::submit,nb::arg("kernel"),nb::arg("buffers"),nb::arg("grid"),
           nb::arg("block"),nb::arg("shared_bytes")=0,nb::call_guard<nb::gil_scoped_release>())
      .def("wait_for_external_stream",&Session::waitForExternalStream,nb::call_guard<nb::gil_scoped_release>())
      .def("check_types",&Session::checkTypes,nb::call_guard<nb::gil_scoped_release>())
      .def("synchronize",&Session::synchronize,nb::call_guard<nb::gil_scoped_release>())
      .def("reap",&Session::reap,nb::call_guard<nb::gil_scoped_release>())
      .def("close",&Session::close,nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro("closed",&Session::closed,nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro("stream",&Session::stream,nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro("device",&Session::device)
      .def_prop_ro("capability",&Session::capability)
      .def_prop_ro("device_name",&Session::deviceName)
      .def_prop_ro("stats",&Session::stats,nb::call_guard<nb::gil_scoped_release>());
}
