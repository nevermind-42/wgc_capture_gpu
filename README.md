# WGC Capture Library

## Problem Statement

Modern applications and games require efficient screen capture with support for:
- 3D graphics (DirectX, OpenGL)
- Protected content (DRM, Netflix)
- HDR content
- High frame rates
- Low latency

Traditional screen capture methods (GDI, PIL, PyAutoGUI) have significant limitations:
- Cannot capture 3D content (shows black screen)
- Do not work with protected content
- Have high latency
- No HDR support
- High CPU usage

## Solution

Windows Graphics Capture (WGC) API provides a modern solution to these problems:
- Direct access to DirectX frame buffer
- Support for all content types
- Minimal latency
- Efficient GPU utilization
- Low CPU usage

## Performance Comparison

![Capture Methods Comparison](capture_methods_comparison.png)

The graph above shows the performance comparison between different capture methods:
- FPS (Frames Per Second)
- Frame capture time
- Frame delays
- Non-empty frame percentage

## Features

- High-performance screen capture
- Real-time frame processing
- Support for multiple monitors
- Low latency capture
- HDR content support
- Protected content support (e.g., Netflix)
- 3D applications and games support

## 3D Content Support

### Windows Graphics Capture (WGC)
- Full DirectX support
- OpenGL applications support
- Games capture support
- Hardware-accelerated video capture
- HDR content support
- Protected content support
- Best performance for 3D applications

### MSS (Multi-Screen-Shot)
- Basic DirectX support
- Limited OpenGL support
- No HDR support
- May have performance issues with 3D content

### GDI (Windows GDI)
- No DirectX support
- No OpenGL support
- No HDR support
- May show black screen for 3D content
- Suitable for static content only

### PyAutoGUI
- No DirectX support
- No OpenGL support
- No HDR support
- May show black screen for 3D content
- Suitable for basic screen capture

### PIL (Python Imaging Library)
- No DirectX support
- No OpenGL support
- No HDR support
- May show black screen for 3D content
- Suitable for static images

## Installation

```bash
pip install wgc-capture
```

## Usage

Basic usage:
```python
import wgc_capture

# Initialize capture
capture = wgc_capture.WGCCapture()

# Get frame
frame = capture.get_frame()
```

## Performance Testing

Run the performance test:
```bash
python test_capture_comparison.py
```

This will test different capture methods and generate a comparison report.

## Performance Metrics

The test measures:
- FPS (Frames Per Second)
- Frame capture time
- Frame delays
- Non-empty frame percentage
- Memory usage

## Requirements

- Windows 10 version 1803 or later
- Python 3.7 or later
- DirectX 11 compatible GPU

## Building from Source

```bash
python setup.py build_ext --inplace
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Python API Methods

### WGCCapture

#### WGCCapture()
Создаёт объект захвата экрана и инициализирует поток захвата.

#### get_frame(timeout_ms=1) -> np.ndarray
Получает текущий кадр как numpy-массив (BGRA, uint8).
- timeout_ms — максимальное время ожидания нового кадра (по умолчанию 1 мс).
- Возвращает: numpy-массив кадра или пустой массив, если кадр не получен.

#### set_frame_callback(callback: Callable[[np.ndarray, dict], None])
Устанавливает callback-функцию, которая будет вызываться при появлении нового кадра.
- callback(frame: np.ndarray, info: dict) — функция, принимающая кадр и метаданные (размер, формат, timestamp и др.).

#### get_monitor_info() -> list[tuple]
Возвращает список информации о доступных мониторах.
- Каждый элемент — кортеж с координатами и размерами монитора.

#### capture_to_cuda_ptr() -> tuple[int, int] | None
Захватывает кадр напрямую в GPU-память (CUDA) и возвращает кортеж:
- cuda_ptr — указатель на память на GPU (целое число, для передачи в PyCUDA и др.).
- pitch — шаг (stride) строки в байтах.
- Если захват не удался, возвращает None.

---

**Пример использования GPU-метода:**
```python
import wgc_capture
import pycuda.driver as cuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np

cap = wgc_capture.WGCCapture()
cuda_ptr, pitch = cap.capture_to_cuda_ptr()
if cuda_ptr is not None:
    width, height = 1920, 1080  # замените на реальные значения
    frame_gpu = gpuarray.GPUArray((height, width, 4), np.uint8, gpudata=cuda.DeviceAllocation(cuda_ptr))
    frame_cpu = frame_gpu.get()
    print("Frame shape:", frame_cpu.shape)
``` 