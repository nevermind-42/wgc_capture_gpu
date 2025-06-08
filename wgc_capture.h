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

using Microsoft::WRL::ComPtr;

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
    WGCCapture(bool use_bgra = true);
    ~WGCCapture();

    // Инициализация и запуск захвата
    bool start_capture();
    void stop_capture();
    
    // Установка callback для новых кадров
    // callback получает указатель на FrameData, который можно безопасно использовать
    // пока не вызван метод release_frame
    void set_frame_callback(std::function<void(FrameData*)> callback);
    
    // Получение информации о мониторе
    std::vector<std::pair<int, int>> get_monitor_info() const;
    
    // Получение кадра
    // Возвращает указатель на FrameData, который нужно освободить через release_frame
    FrameData* get_frame();
    
    // Освобождение кадра (уменьшение счетчика ссылок)
    void release_frame(FrameData* frame);
    
private:
    // Внутренние методы
    void process_frame(cv::Mat& frame, bool is_bgra);
    FrameData* allocate_frame_data();
    
    // Члены класса для Direct3D
    ComPtr<ID3D11Device> d3d_device_;
    ComPtr<ID3D11DeviceContext> d3d_context_;
    winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice d3d_device_winrt_{nullptr};
    
    // Члены класса для Windows Graphics Capture
    winrt::Windows::Graphics::Capture::GraphicsCaptureItem capture_item_{nullptr};
    winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool frame_pool_{nullptr};
    winrt::Windows::Graphics::Capture::GraphicsCaptureSession capture_session_{nullptr};
    
    // Члены класса для управления кадрами
    mutable std::mutex frame_mutex_;
    std::condition_variable frame_cv_;
    
    FrameData current_frame_;  // Текущий кадр (без буферизации)
    bool is_capturing_;
    
    // Callback для новых кадров
    std::function<void(FrameData*)> frame_callback_;
    
    // Информация о мониторе
    std::vector<std::pair<int, int>> monitor_info_;
}; 