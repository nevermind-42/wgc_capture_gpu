#pragma once

#include <windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <wrl/client.h>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <tuple>
#include <atomic>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Graphics.Capture.h>
#include <winrt/Windows.Graphics.DirectX.h>
#include <winrt/Windows.Graphics.DirectX.Direct3D11.h>
#include <windows.graphics.capture.interop.h>
#include <windows.graphics.directx.direct3d11.interop.h>
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>

using Microsoft::WRL::ComPtr;
using namespace winrt;

class WGCCapture {
public:
    WGCCapture();
    ~WGCCapture();

    bool start_capture();
    void stop_capture();
    bool pause_capture();
    bool resume_capture();
    bool is_capturing() const { return is_capturing_; }
    bool is_paused() const { return is_paused_; }

    // Установка callback для новых CUDA-кадров
    // callback получает (void* cuda_ptr, size_t pitch, int width, int height, int64_t timestamp)
    void set_frame_callback(std::function<void(void*, size_t, int, int, int64_t)> callback);

    // Получение информации о мониторах
    std::vector<std::tuple<int, int, int, int>> get_monitor_info() const;

    // Захват кадра напрямую в CUDA память (polling)
    bool capture_to_cuda(void** cuda_ptr, size_t* pitch, int* width, int* height, int64_t* timestamp);

private:
    // Члены класса для Direct3D
    ComPtr<ID3D11Device> d3d_device_;
    ComPtr<ID3D11DeviceContext> d3d_context_;
    winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice d3d_device_winrt_;

    // Члены класса для Windows Graphics Capture
    winrt::Windows::Graphics::Capture::GraphicsCaptureItem capture_item_ = nullptr;
    winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool frame_pool_ = nullptr;
    winrt::Windows::Graphics::Capture::GraphicsCaptureSession capture_session_ = nullptr;

    mutable std::mutex frame_mutex_;
    std::function<void(void*, size_t, int, int, int64_t)> frame_callback_;
    bool is_capturing_ = false;
    bool is_paused_ = false;

    // CUDA ресурсы
    cudaGraphicsResource* cuda_resource_ = nullptr;
    cudaStream_t cuda_stream_ = nullptr;

    // Размеры захвата
    int width_ = 0;
    int height_ = 0;

    // Последний timestamp
    int64_t last_timestamp_ = 0;

    // Последний указатель на texture (для polling)
    ComPtr<ID3D11Texture2D> last_texture_;

    // GPU frame buffer for persistent polling
    void* gpu_frame_buffer_ = nullptr;
    size_t gpu_frame_pitch_ = 0;
    int gpu_frame_width_ = 0;
    int gpu_frame_height_ = 0;

    // Инициализация CUDA-D3D11 интеропа
    bool init_cuda_interop(ID3D11Texture2D* texture);
    void cleanup_cuda_resources();

    // Вспомогательные функции
    static winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice CreateDirect3DDevice(ID3D11Device* d3d_device);
    static ComPtr<ID3D11Device> GetD3D11Device(winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice const& device);

    std::vector<std::tuple<int, int, int, int>> monitor_info_;
}; 