#include "wgc_capture.h"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <thread>
#include <iostream>

// WinRT initialization
struct WinRTInitializer {
    WinRTInitializer() {
        winrt::init_apartment();
    }
    ~WinRTInitializer() {
        winrt::uninit_apartment();
    }
};

// Global WinRT initializer
static WinRTInitializer g_winrt_initializer;

// Helper functions for WinRT and DirectX
namespace {
    // Create IDirect3DDevice from ID3D11Device
    winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice CreateDirect3DDevice(ID3D11Device* d3d_device) {
        ComPtr<IDXGIDevice> dxgi_device;
        HRESULT hr = d3d_device->QueryInterface(IID_PPV_ARGS(&dxgi_device));
        if (FAILED(hr)) {
            std::cerr << "Failed to get DXGI device: " << std::hex << hr << std::endl;
            return nullptr;
        }

        winrt::com_ptr<::IInspectable> inspectable;
        hr = CreateDirect3D11DeviceFromDXGIDevice(dxgi_device.Get(), inspectable.put());
        if (FAILED(hr)) {
            std::cerr << "Failed to create Direct3D11 device from DXGI device: " << std::hex << hr << std::endl;
            return nullptr;
        }

        return inspectable.as<winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice>();
    }

    // Get ID3D11Device from IDirect3DDevice
    ComPtr<ID3D11Device> GetD3D11Device(winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice const& device) {
        auto access = device.as<Windows::Graphics::DirectX::Direct3D11::IDirect3DDxgiInterfaceAccess>();
        ComPtr<ID3D11Device> d3d_device;
        HRESULT hr = access->GetInterface(IID_PPV_ARGS(&d3d_device));
        if (FAILED(hr)) {
            std::cerr << "Failed to get D3D11Device from IDirect3DDevice: " << std::hex << hr << std::endl;
            return nullptr;
        }
        return d3d_device;
    }

    // Get GraphicsCaptureItem for primary monitor
    winrt::Windows::Graphics::Capture::GraphicsCaptureItem CreateCaptureItemForPrimaryMonitor() {
        HMONITOR hmon = MonitorFromWindow(GetDesktopWindow(), MONITOR_DEFAULTTOPRIMARY);
        auto activation_factory = winrt::get_activation_factory<
            winrt::Windows::Graphics::Capture::GraphicsCaptureItem,
            IGraphicsCaptureItemInterop>();
        winrt::Windows::Graphics::Capture::GraphicsCaptureItem item = { nullptr };
        HRESULT hr = activation_factory->CreateForMonitor(
            hmon,
            winrt::guid_of<winrt::Windows::Graphics::Capture::GraphicsCaptureItem>(),
            winrt::put_abi(item));
        if (FAILED(hr)) {
            std::cerr << "Failed to create capture item for monitor: " << std::hex << hr << std::endl;
            return nullptr;
        }
        return item;
    }
}

WGCCapture::WGCCapture(bool use_bgra) : is_capturing_(false) {
    std::cout << "[WGCCapture] Initializing..." << std::endl;
    
    // Инициализация Direct3D
    D3D_FEATURE_LEVEL feature_level;
    HRESULT hr = D3D11CreateDevice(
        nullptr,
        D3D_DRIVER_TYPE_HARDWARE,
        nullptr,
        0,
        nullptr,
        0,
        D3D11_SDK_VERSION,
        d3d_device_.GetAddressOf(),
        &feature_level,
        d3d_context_.GetAddressOf()
    );
    
    if (FAILED(hr)) {
        std::cerr << "[WGCCapture] Failed to create D3D11 device: " << std::hex << hr << std::endl;
        throw std::runtime_error("Failed to create D3D11 device");
    }
    
    std::cout << "[WGCCapture] D3D11 device created with feature level: " << std::hex << feature_level << std::endl;
    
    // Создание WinRT Direct3D устройства
    d3d_device_winrt_ = CreateDirect3DDevice(d3d_device_.Get());
    if (!d3d_device_winrt_) {
        std::cerr << "[WGCCapture] Failed to create WinRT Direct3D device" << std::endl;
        throw std::runtime_error("Failed to create WinRT Direct3D device");
    }
    
    std::cout << "[WGCCapture] WinRT Direct3D device created" << std::endl;
    
    // Получение информации о мониторах
    ComPtr<IDXGIFactory1> dxgi_factory;
    hr = CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)dxgi_factory.GetAddressOf());
    if (FAILED(hr)) {
        std::cerr << "[WGCCapture] Failed to create DXGI factory: " << std::hex << hr << std::endl;
        throw std::runtime_error("Failed to create DXGI factory");
    }
    
    ComPtr<IDXGIAdapter1> adapter;
    for (UINT i = 0; dxgi_factory->EnumAdapters1(i, adapter.GetAddressOf()) != DXGI_ERROR_NOT_FOUND; ++i) {
        ComPtr<IDXGIOutput> output;
        for (UINT j = 0; adapter->EnumOutputs(j, output.GetAddressOf()) != DXGI_ERROR_NOT_FOUND; ++j) {
            DXGI_OUTPUT_DESC output_desc;
            output->GetDesc(&output_desc);
            monitor_info_.push_back({output_desc.DesktopCoordinates.right - output_desc.DesktopCoordinates.left,
                                   output_desc.DesktopCoordinates.bottom - output_desc.DesktopCoordinates.top});
            output.Reset();
        }
        adapter.Reset();
    }
    
    std::cout << "[WGCCapture] Found " << monitor_info_.size() << " monitors" << std::endl;
    std::cout << "[WGCCapture] Initialization complete" << std::endl;
}

WGCCapture::~WGCCapture() {
    stop_capture();
}

bool WGCCapture::start_capture() {
    if (is_capturing_) {
        std::cout << "[WGCCapture] Already capturing" << std::endl;
        return true;
    }
    
    std::cout << "[WGCCapture] Starting capture..." << std::endl;
    
    try {
        // Создаем GraphicsCaptureItem для основного монитора
        capture_item_ = CreateCaptureItemForPrimaryMonitor();
        if (!capture_item_) {
            std::cerr << "[WGCCapture] Failed to create capture item for primary monitor" << std::endl;
            return false;
        }
        
        std::cout << "[WGCCapture] Capture item created for primary monitor" << std::endl;
        
        // Получаем размеры захвата
        auto size = capture_item_.Size();
        std::cout << "[WGCCapture] Capture size: " << size.Width << "x" << size.Height << std::endl;
        
        // Создаем Direct3D11CaptureFramePool
        auto format = winrt::Windows::Graphics::DirectX::DirectXPixelFormat::B8G8R8A8UIntNormalized;
        frame_pool_ = winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool::CreateFreeThreaded(
            d3d_device_winrt_,
            format,
            1,  // Достаточно 1 буфера
            size
        );
        
        // Регистрируем обработчик кадров
        frame_pool_.FrameArrived([this](auto&& sender, auto&&) {
            // Получаем кадр
            auto frame = sender.TryGetNextFrame();
            if (!frame) {
                return;
            }
            
            // Получаем текстуру
            auto surface = frame.Surface();
            auto access = surface.as<Windows::Graphics::DirectX::Direct3D11::IDirect3DDxgiInterfaceAccess>();
            ComPtr<ID3D11Texture2D> texture;
            HRESULT hr = access->GetInterface(IID_PPV_ARGS(&texture));
            if (FAILED(hr)) {
                std::cerr << "[WGCCapture] Failed to get texture from frame: " << std::hex << hr << std::endl;
                return;
            }
            
            // Получаем описание текстуры
            D3D11_TEXTURE2D_DESC desc;
            texture->GetDesc(&desc);
            
            // Создаем текстуру для чтения
            ComPtr<ID3D11Texture2D> staging_texture;
            desc.Usage = D3D11_USAGE_STAGING;
            desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
            desc.BindFlags = 0;
            desc.MiscFlags = 0;
            
            hr = d3d_device_->CreateTexture2D(&desc, nullptr, staging_texture.GetAddressOf());
            if (FAILED(hr)) {
                std::cerr << "[WGCCapture] Failed to create staging texture: " << std::hex << hr << std::endl;
                return;
            }
            
            // Копируем данные
            d3d_context_->CopyResource(staging_texture.Get(), texture.Get());
            
            // Маппим текстуру для чтения
            D3D11_MAPPED_SUBRESOURCE mapped;
            hr = d3d_context_->Map(staging_texture.Get(), 0, D3D11_MAP_READ, 0, &mapped);
            if (FAILED(hr)) {
                std::cerr << "[WGCCapture] Failed to map staging texture: " << std::hex << hr << std::endl;
                return;
            }
            
            // Создаем OpenCV матрицу напрямую из маппированной памяти
            cv::Mat frame_bgra(desc.Height, desc.Width, CV_8UC4, mapped.pData, mapped.RowPitch);
            
            // Обрабатываем кадр напрямую без конвертации цвета
            process_frame(frame_bgra, true);
            
            d3d_context_->Unmap(staging_texture.Get(), 0);
        });
        
        // Создаем сессию захвата
        capture_session_ = frame_pool_.CreateCaptureSession(capture_item_);
        
        // Запускаем захват
        capture_session_.StartCapture();
        is_capturing_ = true;
        
        std::cout << "[WGCCapture] Capture started" << std::endl;
        return true;
    }
    catch (const winrt::hresult_error& e) {
        std::cerr << "[WGCCapture] WinRT error: " << std::hex << e.code() << " - " << winrt::to_string(e.message()) << std::endl;
        return false;
    }
    catch (const std::exception& e) {
        std::cerr << "[WGCCapture] Exception: " << e.what() << std::endl;
        return false;
    }
}

void WGCCapture::stop_capture() {
    if (!is_capturing_) {
        return;
    }
    
    std::cout << "[WGCCapture] Stopping capture..." << std::endl;
    
    try {
        // Останавливаем захват
        if (capture_session_) {
            capture_session_.Close();
            capture_session_ = nullptr;
        }
        
        if (frame_pool_) {
            frame_pool_.Close();
            frame_pool_ = nullptr;
        }
        
        capture_item_ = nullptr;
        is_capturing_ = false;
        
        std::cout << "[WGCCapture] Capture stopped" << std::endl;
    }
    catch (const winrt::hresult_error& e) {
        std::cerr << "[WGCCapture] WinRT error during stop_capture: " << std::hex << e.code() << " - " << winrt::to_string(e.message()) << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "[WGCCapture] Exception during stop_capture: " << e.what() << std::endl;
    }
}

void WGCCapture::process_frame(cv::Mat& frame, bool is_bgra) {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    
    // Сохраняем кадр напрямую
    current_frame_.frame = frame.clone();
    current_frame_.is_bgra = is_bgra;
    current_frame_.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    
    // Подготавливаем информацию о предобработке
    current_frame_.info = std::map<std::string, std::any>{
        {"scale", 1.0},
        {"pad", std::vector<int>{0, 0, 0, 0}},
        {"original_size", std::vector<int>{frame.cols, frame.rows}}
    };
    
    // Увеличиваем счетчик ссылок для использования в callback
    current_frame_.ref_count = 1;
    
    // Вызываем callback если установлен
    if (frame_callback_) {
        frame_callback_(&current_frame_);
    }
    
    frame_cv_.notify_all();
    
    // Уменьшаем счетчик ссылок после использования в callback
    release_frame(&current_frame_);
}

void WGCCapture::set_frame_callback(std::function<void(FrameData*)> callback) {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    frame_callback_ = callback;
}

std::vector<std::pair<int, int>> WGCCapture::get_monitor_info() const {
    return monitor_info_;
}

FrameData* WGCCapture::get_frame() {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    
    if (!is_capturing_) {
        std::cerr << "[WGCCapture] Capture not started" << std::endl;
        return nullptr;
    }
    
    if (current_frame_.frame.empty()) {
        return nullptr;
    }
    
    // Увеличиваем счетчик ссылок и возвращаем текущий кадр
    current_frame_.ref_count++;
    return &current_frame_;
}

void WGCCapture::release_frame(FrameData* frame) {
    if (!frame) return;
    
    std::lock_guard<std::mutex> lock(frame_mutex_);
    
    // Уменьшаем счетчик ссылок
    if (frame->ref_count > 0) {
        frame->ref_count--;
    }
}

FrameData* WGCCapture::allocate_frame_data() {
    return new FrameData();
} 