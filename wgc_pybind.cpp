#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <memory>
#include <iostream>
#include "wgc_capture.h"

namespace py = pybind11;

class PyWGCCapture {
public:
    PyWGCCapture() = default;
    ~PyWGCCapture() = default;

    bool start_capture() { return capture_.start_capture(); }
    void stop_capture() { capture_.stop_capture(); }
    bool pause_capture() { return capture_.pause_capture(); }
    bool resume_capture() { return capture_.resume_capture(); }
    bool is_capturing() const { return capture_.is_capturing(); }
    bool is_paused() const { return capture_.is_paused(); }
    
    std::vector<std::tuple<int, int, int, int>> get_monitor_info() const {
        return capture_.get_monitor_info();
    }

    void set_frame_callback(py::function callback) {
        capture_.set_frame_callback([callback](void* data, size_t pitch, int width, int height, int64_t timestamp) {
            py::gil_scoped_acquire gil;
            callback(py::cast(data), py::cast(pitch), py::cast(width), py::cast(height), py::cast(timestamp));
        });
    }

    std::tuple<py::object, size_t, int, int, int64_t> capture_to_cuda() {
        void* cuda_ptr = nullptr;
        size_t pitch = 0;
        int width = 0, height = 0;
        int64_t timestamp = 0;

        if (capture_.capture_to_cuda(&cuda_ptr, &pitch, &width, &height, &timestamp)) {
            return std::make_tuple(
                py::cast(cuda_ptr),
                pitch,
                width,
                height,
                timestamp
            );
        }
        return std::make_tuple(py::none(), 0, 0, 0, 0);
    }

    py::tuple capture_to_cuda_ptr() {
        void* cuda_ptr = nullptr;
        size_t pitch = 0;
        int width = 0, height = 0;
        int64_t ts = 0;
        bool ok = capture_.capture_to_cuda(&cuda_ptr, &pitch, &width, &height, &ts);
        if (!ok) return py::none();
        return py::make_tuple(reinterpret_cast<uintptr_t>(cuda_ptr), pitch, width, height, ts);
    }

private:
    WGCCapture capture_;
};

PYBIND11_MODULE(wgc_capture, m) {
    m.doc() = "Windows Graphics Capture module (CUDA only)";
    py::class_<PyWGCCapture>(m, "WGCCapture")
        .def(py::init<>())
        .def("start_capture", &PyWGCCapture::start_capture)
        .def("stop_capture", &PyWGCCapture::stop_capture)
        .def("pause_capture", &PyWGCCapture::pause_capture)
        .def("resume_capture", &PyWGCCapture::resume_capture)
        .def("is_capturing", &PyWGCCapture::is_capturing)
        .def("is_paused", &PyWGCCapture::is_paused)
        .def("get_monitor_info", &PyWGCCapture::get_monitor_info)
        .def("set_frame_callback", &PyWGCCapture::set_frame_callback)
        .def("capture_to_cuda", &PyWGCCapture::capture_to_cuda)
        .def("capture_to_cuda_ptr", &PyWGCCapture::capture_to_cuda_ptr);
} 