#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "wgc_capture.h"
#include <memory>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <thread>

namespace py = pybind11;

// Use pybind11::ssize_t type
using ssize_t = py::ssize_t;

namespace {
    bool debug_mode = true;
}

class PyWGCCapture {
public:
    PyWGCCapture() : cap(new WGCCapture()) {
        if (debug_mode) std::cerr << "[PyWGCCapture] Initializing WGCCapture..." << std::endl;
        if (!cap->start_capture()) {
            if (debug_mode) std::cerr << "[PyWGCCapture] WGCCapture::start_capture() failed!" << std::endl;
            throw std::runtime_error("Failed to initialize WGCCapture");
        }
        if (debug_mode) std::cerr << "[PyWGCCapture] WGCCapture initialized successfully." << std::endl;
    }
    
    ~PyWGCCapture() {
        if (debug_mode) std::cerr << "[PyWGCCapture] Cleanup WGCCapture..." << std::endl;
        cap->stop_capture();
    }

    py::array get_frame(int timeout_ms = 1) {
        if (debug_mode) std::cerr << "[PyWGCCapture] Calling get_frame..." << std::endl;
        try {
            FrameData* frame_data = cap->get_frame();
            if (!frame_data || frame_data->frame.empty()) {
                if (debug_mode) std::cerr << "[PyWGCCapture] No frame available, returning empty array" << std::endl;
                return py::array_t<uint8_t>();
            }
            cv::Mat& frame = frame_data->frame;
            std::vector<ssize_t> shape = {frame.rows, frame.cols, frame.channels()};
            std::vector<ssize_t> strides = {static_cast<ssize_t>(frame.step[0]), static_cast<ssize_t>(frame.step[1]), static_cast<ssize_t>(sizeof(uint8_t))};
            // Return a copy of the data
            auto result = py::array_t<uint8_t>(shape, strides, frame.data).attr("copy")();
            if (debug_mode) std::cerr << "[PyWGCCapture] Frame received: " << frame.cols << "x" << frame.rows << std::endl;
            return result;
        } catch (const std::exception& e) {
            if (debug_mode) std::cerr << "[PyWGCCapture] Exception in get_frame: " << e.what() << std::endl;
            throw std::runtime_error("Failed to get frame");
        }
    }

    // Set callback for new frames with direct texture access
    void set_frame_callback(py::function callback) {
        cap->set_frame_callback([callback, this](FrameData* frame_data) {
            if (debug_mode) std::cerr << "[PyWGCCapture] C++: frame_callback called, frame=" << (frame_data ? "OK" : "nullptr") << std::endl;
            py::gil_scoped_acquire gil;
            if (!frame_data || frame_data->frame.empty()) {
                std::cerr << "[PyWGCCapture] C++: frame_data is empty in callback" << std::endl;
                return;
            }
            cv::Mat& frame = frame_data->frame;
            std::vector<ssize_t> shape = {frame.rows, frame.cols, frame.channels()};
            std::vector<ssize_t> strides = {static_cast<ssize_t>(frame.step[0]), static_cast<ssize_t>(frame.step[1]), static_cast<ssize_t>(sizeof(uint8_t))};
            // Pass a copy of the data
            auto array = py::array_t<uint8_t>(shape, strides, frame.data).attr("copy")();
            py::dict py_info;
            if (auto it = frame_data->info.find("original_size"); it != frame_data->info.end()) {
                try {
                    auto size = std::any_cast<std::vector<int>>(it->second);
                    py_info["original_size"] = py::cast(size);
                } catch (const std::bad_any_cast&) {
                    std::cerr << "[PyWGCCapture] C++: Failed to cast 'original_size'" << std::endl;
                }
            }
            py_info["is_bgra"] = frame_data->is_bgra;
            py_info["timestamp"] = frame_data->timestamp;
            
            // Add D3D11 texture info for direct GPU processing
            TextureResource texture = cap->get_texture_resource();
            py_info["texture_ptr"] = reinterpret_cast<uint64_t>(texture.texture);
            py_info["texture_width"] = texture.width;
            py_info["texture_height"] = texture.height;
            py_info["texture_pitch"] = texture.pitch;
            py_info["d3d11_device_ptr"] = reinterpret_cast<uint64_t>(cap->get_d3d11_device());
            
            try {
                std::cerr << "[PyWGCCapture] C++: calling Python callback..." << std::endl;
                callback(array, py_info);
                cap->release_frame(frame_data);
                std::cerr << "[PyWGCCapture] C++: Python callback completed" << std::endl;
            } catch (const py::error_already_set& e) {
                std::cerr << "[PyWGCCapture] C++: Python error in callback: " << e.what() << std::endl;
            }
        });
    }

    // Новый метод колбэка с прямой передачей текстуры D3D11
    void set_texture_callback(py::function callback) {
        if (debug_mode) std::cerr << "[PyWGCCapture] Setting texture callback..." << std::endl;
        
        cap->set_texture_callback([callback, this](FrameCallbackData* data) {
            if (debug_mode) std::cerr << "[PyWGCCapture] C++: texture_callback called, texture=" 
                << (data && data->texture ? "OK" : "nullptr") << std::endl;
            
            py::gil_scoped_acquire gil;
            
            if (!data || !data->texture) {
                std::cerr << "[PyWGCCapture] C++: texture_data is nullptr in callback" << std::endl;
                return;
            }
            
            // Создаем словарь с информацией о текстуре для передачи в Python
            py::dict texture_info;
            texture_info["texture_ptr"] = reinterpret_cast<uint64_t>(data->texture);
            texture_info["width"] = data->width;
            texture_info["height"] = data->height;
            texture_info["timestamp"] = data->timestamp;
            texture_info["d3d11_device_ptr"] = reinterpret_cast<uint64_t>(cap->get_d3d11_device());
            
            try {
                if (debug_mode) std::cerr << "[PyWGCCapture] C++: calling Python texture callback..." << std::endl;
                callback(texture_info);
                if (debug_mode) std::cerr << "[PyWGCCapture] C++: Python texture callback completed" << std::endl;
            } catch (const py::error_already_set& e) {
                std::cerr << "[PyWGCCapture] C++: Python error in texture callback: " << e.what() << std::endl;
            }
        });
    }

    // Get monitor information
    py::list get_monitor_info() {
        auto info = cap->get_monitor_info();
        py::list result;
        for (const auto& [width, height] : info) {
            result.append(py::make_tuple(width, height));
        }
        return result;
    }
    
    // Новые методы для работы с D3D11 текстурой
    
    // Проверить, есть ли новая текстура
    bool has_new_texture() {
        return cap->has_new_texture();
    }
    
    // Получить размеры текстуры
    py::tuple get_texture_size() {
        auto [width, height] = cap->get_texture_size();
        return py::make_tuple(width, height);
    }
    
    // Получить указатель на D3D11 устройство
    uint64_t get_d3d11_device_ptr() {
        return reinterpret_cast<uint64_t>(cap->get_d3d11_device());
    }
    
    // Получить информацию о текстуре для использования в PyCUDA
    py::dict get_texture_info() {
        TextureResource resource = cap->get_texture_resource();
        
        py::dict result;
        if (!resource.texture) {
            return result;
        }
        
        result["texture_ptr"] = reinterpret_cast<uint64_t>(resource.texture);
        result["width"] = resource.width;
        result["height"] = resource.height;
        result["pitch"] = resource.pitch;
        result["timestamp"] = resource.timestamp;
        
        return result;
    }

private:
    std::unique_ptr<WGCCapture> cap;
};

void set_debug(bool value) { debug_mode = value; }

PYBIND11_MODULE(wgc_capture, m) {
    m.doc() = "Windows Graphics Capture module"; 

    py::class_<PyWGCCapture>(m, "WGCCapture")
        .def(py::init<>())
        .def("set_frame_callback", &PyWGCCapture::set_frame_callback)
        .def("set_texture_callback", &PyWGCCapture::set_texture_callback)
        .def("get_monitor_info", &PyWGCCapture::get_monitor_info)
        .def("get_frame", &PyWGCCapture::get_frame, py::arg("timeout_ms") = 1)
        .def("has_new_texture", &PyWGCCapture::has_new_texture)
        .def("get_texture_size", &PyWGCCapture::get_texture_size)
        .def("get_d3d11_device_ptr", &PyWGCCapture::get_d3d11_device_ptr)
        .def("get_texture_info", &PyWGCCapture::get_texture_info);
    
    m.def("set_debug", &set_debug, "Enable or disable debug output");
} 