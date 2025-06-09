#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "wgc_capture.h"
#include <opencv2/opencv.hpp>

namespace py = pybind11;

PYBIND11_MODULE(wgc_capture, m) {
    py::class_<WGCCapture>(m, "WGCCapture")
        .def(py::init<bool>(), py::arg("use_bgra") = false)
        .def("start_capture", &WGCCapture::start_capture)
        .def("stop_capture", &WGCCapture::stop_capture)
        .def("get_frame", [](WGCCapture& self) -> py::array_t<uint8_t> {
            FrameData* frame_data = self.get_frame();
            if (!frame_data || frame_data->frame.empty())
                return py::array_t<uint8_t>();
            cv::Mat& frame = frame_data->frame;
            py::array_t<uint8_t> result(
                {frame.rows, frame.cols, frame.channels()},
                {frame.step[0], frame.step[1], sizeof(uint8_t)},
                frame.data
            );
            self.release_frame(frame_data);
            return result;
        })
        .def("get_monitor_info", &WGCCapture::get_monitor_info)
        .def("set_frame_callback", &WGCCapture::set_frame_callback);
    m.doc() = "Windows Graphics Capture module with CUDA support";
} 