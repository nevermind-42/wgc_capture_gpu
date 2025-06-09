#pragma once

#include <windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <wrl/client.h>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <map>
#include <any>
#include <tuple>
#include <atomic>
#include <opencv2/opencv.hpp>

// Включения для Windows Graphics Capture API
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Graphics.Capture.h>
#include <winrt/Windows.Graphics.DirectX.h>
#include <winrt/Windows.Graphics.DirectX.Direct3D11.h>
#include <windows.graphics.capture.interop.h>
#include <windows.graphics.directx.direct3d11.interop.h>

#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#include <pybind11/numpy.h>

using Microsoft::WRL::ComPtr;
using namespace winrt;

// Структура для хранения кадра и его метаданных
struct FrameData {
    cv::Mat frame;                           // Кадр (BGRA формат)
    std::map<std::string, std::any> info;    // Метаданные кадра
    int64_t timestamp;                       // Временная метка кадра
    bool is_bgra;                            // Флаг формата (всегда true для BGRA)
    int ref_count;                           // Счетчик ссылок для отслеживания использования

    FrameData() : timestamp(0), is_bgra(true), ref_count(0) {}
};

class WGCCapture {
public:
    WGCCapture(bool use_bgra = false);
    ~WGCCapture();

    // Инициализация и запуск захвата
    bool start_capture();
    void stop_capture();
    bool is_capturing() const { return is_capturing_; }
    
    // Установка callback для новых кадров
    // callback получает указатель на FrameData, который можно безопасно использовать
    // пока не вызван метод release_frame
    void set_frame_callback(std::function<void(FrameData*)> callback);
    
    // Получение информации о мониторах
    std::vector<std::tuple<int, int, int, int>> get_monitor_info() const;
    
    // Получение кадра
    // Возвращает указатель на FrameData, который нужно освободить через release_frame
    FrameData* get_frame();
    
    // Освобождение кадра (уменьшение счетчика ссылок)
    void release_frame(FrameData* frame);
    
    // Захват кадра напрямую в CUDA память
    bool capture_to_cuda(void** cuda_ptr, size_t* pitch);
    
private:
    // Внутренние методы
    void process_frame(cv::Mat& frame, bool is_bgra);
    FrameData* allocate_frame_data();
    
    // Члены класса для Direct3D
    ComPtr<ID3D11Device> d3d_device_;
    ComPtr<ID3D11DeviceContext> d3d_context_;
    winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice d3d_device_winrt_;
    
    // Члены класса для Windows Graphics Capture
    winrt::Windows::Graphics::Capture::GraphicsCaptureItem capture_item_ = nullptr;
    winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool frame_pool_ = nullptr;
    winrt::Windows::Graphics::Capture::GraphicsCaptureSession capture_session_ = nullptr;
    
    // Члены класса для управления кадрами
    mutable std::mutex frame_mutex_;
    std::condition_variable frame_cv_;
    
    FrameData current_frame_;  // Текущий кадр (без буферизации)
    bool is_capturing_;
    
    // Callback для новых кадров
    std::function<void(FrameData*)> frame_callback_;
    
    // CUDA ресурсы
    cudaGraphicsResource* cuda_resource_ = nullptr;
    cudaStream_t cuda_stream_ = nullptr;
    
    // Размеры захвата
    int width_ = 0;
    int height_ = 0;
    
    // Инициализация CUDA-D3D11 интеропа
    bool init_cuda_interop(ID3D11Texture2D* texture);
    void cleanup_cuda_resources();
    
    // Вспомогательные функции
    static winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice CreateDirect3DDevice(ID3D11Device* d3d_device);
    static ComPtr<ID3D11Device> GetD3D11Device(winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice const& device);

    std::vector<std::tuple<int, int, int, int>> monitor_info_;
}; 