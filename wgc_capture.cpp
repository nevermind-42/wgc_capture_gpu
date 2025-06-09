#include "wgc_capture.h"
#include <iostream>
#include "dxgi_access.h"

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
    // Get ID3D11Device from IDirect3DDevice
    ComPtr<ID3D11Device> GetD3D11Device(winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice const& device) {
        winrt::com_ptr<IUnknown> unk = device.as<IUnknown>();
        ComPtr<IDirect3DDxgiInterfaceAccess> dxgi_access;
        unk->QueryInterface(__uuidof(IDirect3DDxgiInterfaceAccess), reinterpret_cast<void**>(dxgi_access.GetAddressOf()));
        ComPtr<ID3D11Device> d3d_device;
        HRESULT hr = dxgi_access->GetInterface(IID_PPV_ARGS(&d3d_device));
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
        winrt::Windows::Graphics::Capture::GraphicsCaptureItem item = nullptr;
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

WGCCapture::WGCCapture() 
    : is_capturing_(false)
    , is_paused_(false)
    , cuda_resource_(nullptr)
    , cuda_stream_(nullptr)
    , width_(0)
    , height_(0)
    , last_timestamp_(0)
    , gpu_frame_buffer_(nullptr)
    , gpu_frame_pitch_(0)
    , gpu_frame_width_(0)
    , gpu_frame_height_(0)
{
    std::cout << "[WGCCapture-DEBUG] WGCCapture constructor" << std::endl;
    std::cout << "[WGCCapture-DEBUG] Initializing Direct3D..." << std::endl;
    
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
        std::cerr << "[WGCCapture-DEBUG] Failed to create D3D11 device: " << std::hex << hr << std::endl;
        throw std::runtime_error("Failed to create D3D11 device");
    }
    
    std::cout << "[WGCCapture-DEBUG] D3D11 device created, feature level: " << std::hex << feature_level << std::endl;
    
    // Создание WinRT Direct3D устройства
    d3d_device_winrt_ = CreateDirect3DDevice(d3d_device_.Get());
    if (!d3d_device_winrt_) {
        std::cerr << "[WGCCapture-DEBUG] Failed to create WinRT Direct3D device" << std::endl;
        throw std::runtime_error("Failed to create WinRT Direct3D device");
    }
    
    std::cout << "[WGCCapture-DEBUG] WinRT Direct3D device created" << std::endl;
    
    // Получение информации о мониторах
    ComPtr<IDXGIFactory1> dxgi_factory;
    hr = CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)dxgi_factory.GetAddressOf());
    if (FAILED(hr)) {
        std::cerr << "[WGCCapture-DEBUG] Failed to create DXGI factory: " << std::hex << hr << std::endl;
        throw std::runtime_error("Failed to create DXGI factory");
    }
    
    ComPtr<IDXGIAdapter1> adapter;
    for (UINT i = 0; dxgi_factory->EnumAdapters1(i, adapter.GetAddressOf()) != DXGI_ERROR_NOT_FOUND; ++i) {
        ComPtr<IDXGIOutput> output;
        for (UINT j = 0; adapter->EnumOutputs(j, output.GetAddressOf()) != DXGI_ERROR_NOT_FOUND; ++j) {
            DXGI_OUTPUT_DESC output_desc;
            output->GetDesc(&output_desc);
            monitor_info_.push_back(std::make_tuple(
                output_desc.DesktopCoordinates.left,
                output_desc.DesktopCoordinates.top,
                output_desc.DesktopCoordinates.right,
                output_desc.DesktopCoordinates.bottom
            ));
            output.Reset();
        }
        adapter.Reset();
    }
    
    std::cout << "[WGCCapture-DEBUG] Monitors found: " << monitor_info_.size() << std::endl;
    std::cout << "[WGCCapture-DEBUG] Initialization complete" << std::endl;

    // Создаем CUDA поток
    cudaError_t err = cudaStreamCreate(&cuda_stream_);
    if (err != cudaSuccess) {
        std::cerr << "[WGCCapture-DEBUG] Failed to create CUDA stream: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("Failed to create CUDA stream");
    }
    std::cout << "[WGCCapture-DEBUG] CUDA stream created" << std::endl;

    // Добавляем поля для GPU-буфера
    gpu_frame_buffer_ = nullptr;
    gpu_frame_pitch_ = 0;
    gpu_frame_width_ = 0;
    gpu_frame_height_ = 0;
}

WGCCapture::~WGCCapture() {
    stop_capture();
    cleanup_cuda_resources();
}

void WGCCapture::cleanup_cuda_resources() {
    std::cout << "[WGCCapture-DEBUG] cleanup_cuda_resources called" << std::endl;
    
    if (cuda_resource_) {
        std::cout << "[WGCCapture-DEBUG] Releasing cuda_resource_..." << std::endl;
        cudaGraphicsUnregisterResource(cuda_resource_);
        cuda_resource_ = nullptr;
    }

    if (cuda_stream_) {
        std::cout << "[WGCCapture-DEBUG] Releasing cuda_stream_..." << std::endl;
        cudaStreamDestroy(cuda_stream_);
        cuda_stream_ = nullptr;
    }

    if (gpu_frame_buffer_) {
        std::cout << "[WGCCapture-DEBUG] Freeing gpu_frame_buffer_..." << std::endl;
        cudaFree(gpu_frame_buffer_);
        gpu_frame_buffer_ = nullptr;
        gpu_frame_pitch_ = 0;
        gpu_frame_width_ = 0;
        gpu_frame_height_ = 0;
    }
}

bool WGCCapture::init_cuda_interop(ID3D11Texture2D* texture) {
    std::cout << "[WGCCapture-DEBUG] init_cuda_interop called, texture=" << texture << std::endl;
    if (!texture) {
        std::cerr << "[WGCCapture-DEBUG] texture == nullptr!" << std::endl;
        return false;
    }

    // Получаем описание текстуры
    D3D11_TEXTURE2D_DESC desc;
    texture->GetDesc(&desc);
    
    // Проверяем, что текстура поддерживает CUDA
    if (desc.Usage != D3D11_USAGE_DEFAULT) {
        std::cerr << "[WGCCapture-DEBUG] Texture usage is not D3D11_USAGE_DEFAULT" << std::endl;
        return false;
    }

    // Освобождаем предыдущий ресурс, если он существует
    if (cuda_resource_) {
        std::cout << "[WGCCapture-DEBUG] Releasing previous CUDA resource..." << std::endl;
        cudaGraphicsUnregisterResource(cuda_resource_);
        cuda_resource_ = nullptr;
    }

    // Регистрируем новый ресурс
    cudaError_t err = cudaGraphicsD3D11RegisterResource(
        &cuda_resource_,
        texture,
        cudaGraphicsRegisterFlagsNone
    );

    if (err != cudaSuccess) {
        std::cerr << "[WGCCapture-DEBUG] Failed to register CUDA resource: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    // Выделяем GPU буфер, если его еще нет или размер изменился
    if (!gpu_frame_buffer_ || gpu_frame_width_ != desc.Width || gpu_frame_height_ != desc.Height) {
        if (gpu_frame_buffer_) {
            std::cout << "[WGCCapture-DEBUG] Reallocating GPU frame buffer..." << std::endl;
            cudaFree(gpu_frame_buffer_);
        }

        size_t pitch;
        err = cudaMallocPitch(&gpu_frame_buffer_, &pitch, desc.Width * 4, desc.Height);
        if (err != cudaSuccess) {
            std::cerr << "[WGCCapture-DEBUG] Failed to allocate GPU frame buffer: " << cudaGetErrorString(err) << std::endl;
            return false;
        }

        gpu_frame_pitch_ = pitch;
        gpu_frame_width_ = desc.Width;
        gpu_frame_height_ = desc.Height;
        std::cout << "[WGCCapture-DEBUG] GPU frame buffer allocated: " << gpu_frame_width_ << "x" << gpu_frame_height_ << std::endl;
    }

    std::cout << "[WGCCapture-DEBUG] CUDA resource registered successfully: " << cuda_resource_ << std::endl;
    return true;
}

bool WGCCapture::capture_to_cuda(void** cuda_ptr, size_t* pitch, int* width, int* height, int64_t* timestamp) {
    std::cout << "[WGCCapture-DEBUG] capture_to_cuda called" << std::endl;
    std::lock_guard<std::mutex> lock(frame_mutex_);
    
    if (!is_capturing_ || !gpu_frame_buffer_) {
        std::cerr << "[WGCCapture-DEBUG] No active capture or gpu_frame_buffer_ == nullptr" << std::endl;
        return false;
    }

    if (!last_texture_) {
        std::cerr << "[WGCCapture-DEBUG] No texture available" << std::endl;
        return false;
    }

    // Инициализируем CUDA ресурс для текущей текстуры
    if (!init_cuda_interop(last_texture_.Get())) {
        std::cerr << "[WGCCapture-DEBUG] Failed to init CUDA interop for frame copy!" << std::endl;
        return false;
    }

    // Копируем данные из текстуры в GPU буфер
    cudaError_t err = cudaGraphicsMapResources(1, &cuda_resource_, cuda_stream_);
    if (err != cudaSuccess) {
        std::cerr << "[WGCCapture-DEBUG] Failed to map CUDA resource: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    cudaArray_t cuda_array;
    err = cudaGraphicsSubResourceGetMappedArray(&cuda_array, cuda_resource_, 0, 0);
    if (err != cudaSuccess) {
        std::cerr << "[WGCCapture-DEBUG] Failed to get mapped array: " << cudaGetErrorString(err) << std::endl;
        cudaGraphicsUnmapResources(1, &cuda_resource_, cuda_stream_);
        return false;
    }

    err = cudaMemcpy2DAsync(
        gpu_frame_buffer_, gpu_frame_pitch_,
        cuda_array, 0,
        gpu_frame_width_ * 4, gpu_frame_height_,
        cudaMemcpyDeviceToDevice,
        cuda_stream_
    );

    cudaGraphicsUnmapResources(1, &cuda_resource_, cuda_stream_);

    if (err != cudaSuccess) {
        std::cerr << "[WGCCapture-DEBUG] Failed to copy frame to GPU buffer: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    // Синхронизируем поток
    cudaStreamSynchronize(cuda_stream_);

    *cuda_ptr = gpu_frame_buffer_;
    if (pitch) *pitch = gpu_frame_pitch_;
    if (width) *width = gpu_frame_width_;
    if (height) *height = gpu_frame_height_;
    if (timestamp) *timestamp = last_timestamp_;

    std::cout << "[WGCCapture-DEBUG] Frame copied to GPU buffer successfully" << std::endl;
    return true;
}

bool WGCCapture::start_capture() {
    std::cout << "[WGCCapture-DEBUG] start_capture called" << std::endl;
    if (is_capturing_) {
        std::cout << "[WGCCapture-DEBUG] Already capturing" << std::endl;
        return true;
    }
    if (is_paused_) {
        return resume_capture();
    }
    std::cout << "[WGCCapture-DEBUG] Starting capture..." << std::endl;
    try {
        capture_item_ = CreateCaptureItemForPrimaryMonitor();
        if (!capture_item_) {
            std::cerr << "[WGCCapture-DEBUG] Failed to create capture_item_ for primary monitor" << std::endl;
            return false;
        }
        std::cout << "[WGCCapture-DEBUG] capture_item_ created: " << (capture_item_ ? "ok" : "nullptr") << std::endl;
        auto size = capture_item_.Size();
        std::cout << "[WGCCapture-DEBUG] Capture size: " << size.Width << "x" << size.Height << std::endl;
        auto format = winrt::Windows::Graphics::DirectX::DirectXPixelFormat::B8G8R8A8UIntNormalized;
        frame_pool_ = winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool::CreateFreeThreaded(
            d3d_device_winrt_,
            format,
            1,
            size
        );
        std::cout << "[WGCCapture-DEBUG] frame_pool_ created" << std::endl;
        frame_pool_.FrameArrived([this](auto&& sender, auto&&) {
            std::cout << "[WGCCapture-DEBUG] FrameArrived callback!" << std::endl;
            auto frame = sender.TryGetNextFrame();
            if (!frame) {
                std::cerr << "[WGCCapture-DEBUG] frame == nullptr" << std::endl;
                return;
            }
            auto surface = frame.Surface();
            winrt::com_ptr<IUnknown> surface_unk = surface.as<IUnknown>();
            ComPtr<IDirect3DDxgiInterfaceAccess> dxgi_access;
            surface_unk->QueryInterface(__uuidof(IDirect3DDxgiInterfaceAccess), reinterpret_cast<void**>(dxgi_access.GetAddressOf()));
            ComPtr<ID3D11Texture2D> texture;
            HRESULT hr = dxgi_access->GetInterface(IID_PPV_ARGS(&texture));
            if (FAILED(hr)) {
                std::cerr << "[WGCCapture-DEBUG] Failed to get ID3D11Texture2D: " << std::hex << hr << std::endl;
                return;
            }
            D3D11_TEXTURE2D_DESC desc;
            texture->GetDesc(&desc);
            width_ = desc.Width;
            height_ = desc.Height;
            last_texture_ = texture;
            last_timestamp_ = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            std::cout << "[WGCCapture-DEBUG] New frame: texture=" << texture.Get() << ", size=" << width_ << "x" << height_ << ", ts=" << last_timestamp_ << std::endl;
            // --- GPU буфер ---
            if (!gpu_frame_buffer_ || gpu_frame_width_ != width_ || gpu_frame_height_ != height_) {
                if (gpu_frame_buffer_) {
                    std::cout << "[WGCCapture-DEBUG] Reallocating gpu_frame_buffer_ for new size..." << std::endl;
                    cudaFree(gpu_frame_buffer_);
                }
                size_t buf_size = width_ * height_ * 4;
                cudaError_t err = cudaMalloc(&gpu_frame_buffer_, buf_size);
                if (err != cudaSuccess) {
                    std::cerr << "[WGCCapture-DEBUG] Failed to allocate gpu_frame_buffer_: " << cudaGetErrorString(err) << std::endl;
                    gpu_frame_buffer_ = nullptr;
                    return;
                }
                gpu_frame_width_ = width_;
                gpu_frame_height_ = height_;
                gpu_frame_pitch_ = width_ * 4;
                std::cout << "[WGCCapture-DEBUG] Allocated gpu_frame_buffer_ " << gpu_frame_buffer_ << " size=" << buf_size << std::endl;
            }
            // Копируем кадр в GPU буфер
            if (init_cuda_interop(texture.Get())) {
                void* src_ptr = nullptr;
                size_t src_pitch = 0;
                cudaGraphicsMapResources(1, &cuda_resource_, cuda_stream_);
                cudaGraphicsResourceGetMappedPointer(&src_ptr, &src_pitch, cuda_resource_);
                cudaMemcpy2DAsync(gpu_frame_buffer_, gpu_frame_pitch_, src_ptr, src_pitch, width_ * 4, height_, cudaMemcpyDeviceToDevice, cuda_stream_);
                cudaGraphicsUnmapResources(1, &cuda_resource_, cuda_stream_);
                cudaStreamSynchronize(cuda_stream_);
                std::cout << "[WGCCapture-DEBUG] Copied frame to gpu_frame_buffer_=" << gpu_frame_buffer_ << std::endl;
            } else {
                std::cerr << "[WGCCapture-DEBUG] Failed to init CUDA interop for frame copy!" << std::endl;
            }
            // --- конец GPU буфера ---
            if (frame_callback_) {
                std::cout << "[WGCCapture-DEBUG] Calling frame_callback_..." << std::endl;
                frame_callback_(gpu_frame_buffer_, gpu_frame_pitch_, width_, height_, last_timestamp_);
            }
        });
        capture_session_ = frame_pool_.CreateCaptureSession(capture_item_);
        std::cout << "[WGCCapture-DEBUG] capture_session_ created" << std::endl;
        capture_session_.StartCapture();
        is_capturing_ = true;
        is_paused_ = false;
        std::cout << "[WGCCapture-DEBUG] Capture started successfully" << std::endl;
        return true;
    }
    catch (const winrt::hresult_error& e) {
        std::cerr << "[WGCCapture-DEBUG] WinRT error: " << winrt::to_string(e.message()) << std::endl;
        return false;
    }
}

void WGCCapture::stop_capture() {
    std::cout << "[WGCCapture-DEBUG] stop_capture called" << std::endl;
    std::lock_guard<std::mutex> lock(frame_mutex_);
    if (!is_capturing_) {
        std::cout << "[WGCCapture-DEBUG] Not capturing" << std::endl;
        return;
    }
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
    is_paused_ = false;
    std::cout << "[WGCCapture-DEBUG] Capture stopped" << std::endl;
}

bool WGCCapture::pause_capture() {
    std::cout << "[WGCCapture-DEBUG] pause_capture called" << std::endl;
    std::lock_guard<std::mutex> lock(frame_mutex_);
    if (!is_capturing_) {
        std::cout << "[WGCCapture-DEBUG] Not capturing" << std::endl;
        return false;
    }
    if (is_paused_) {
        std::cout << "[WGCCapture-DEBUG] Already paused" << std::endl;
        return true;
    }
    if (capture_session_) {
        capture_session_.Close();
        capture_session_ = nullptr;
        is_paused_ = true;
        std::cout << "[WGCCapture-DEBUG] Capture paused" << std::endl;
        return true;
    }
    return false;
}

bool WGCCapture::resume_capture() {
    std::cout << "[WGCCapture-DEBUG] resume_capture called" << std::endl;
    std::lock_guard<std::mutex> lock(frame_mutex_);
    if (!is_capturing_) {
        std::cout << "[WGCCapture-DEBUG] Not capturing" << std::endl;
        return false;
    }
    if (!is_paused_) {
        std::cout << "[WGCCapture-DEBUG] Not paused" << std::endl;
        return true;
    }
    
    // Создаем новую сессию захвата
    if (capture_item_ && frame_pool_) {
        capture_session_ = frame_pool_.CreateCaptureSession(capture_item_);
        capture_session_.StartCapture();
        is_paused_ = false;
        std::cout << "[WGCCapture-DEBUG] Capture resumed" << std::endl;
        return true;
    }
    return false;
}

std::vector<std::tuple<int, int, int, int>> WGCCapture::get_monitor_info() const {
    return monitor_info_;
}

winrt::Windows::Graphics::Capture::GraphicsCaptureItem capture_item_ = nullptr;
winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool frame_pool_ = nullptr;
winrt::Windows::Graphics::Capture::GraphicsCaptureSession capture_session_ = nullptr;

winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice WGCCapture::CreateDirect3DDevice(ID3D11Device* d3d_device) {
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

void WGCCapture::set_frame_callback(std::function<void(void*, size_t, int, int, int64_t)> callback) {
    std::cout << "[WGCCapture-DEBUG] set_frame_callback called" << std::endl;
    std::lock_guard<std::mutex> lock(frame_mutex_);
    frame_callback_ = std::move(callback);
    std::cout << "[WGCCapture-DEBUG] frame_callback_ set" << std::endl;
} 