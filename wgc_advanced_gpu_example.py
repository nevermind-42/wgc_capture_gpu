import wgc_capture
import numpy as np
import cv2
import time

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    import pycuda.gpuarray as gpuarray
    from pycuda.compiler import SourceModule
    HAS_CUDA = True
except ImportError:
    print("PyCUDA не установлен, требуется для выполнения этого примера")
    HAS_CUDA = False
    exit(1)

# CUDA-ядро для обработки кадра на GPU (инверсия цвета и добавление эффектов)
cuda_code = """
__global__ void process_frame(unsigned char* input, unsigned char* output, 
                              int width, int height, int pitch, float time) {
    // Определяем координаты пикселя
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Проверка границ
    if (x >= width || y >= height)
        return;
    
    // Индекс в буфере (используем pitch для правильного выравнивания)
    const int input_idx = y * pitch + x * 4;
    const int output_idx = y * width * 4 + x * 4;
    
    // Получаем RGB компоненты
    unsigned char b = input[input_idx];
    unsigned char g = input[input_idx + 1];
    unsigned char r = input[input_idx + 2];
    unsigned char a = input[input_idx + 3];
    
    // Создаем волновой эффект на основе времени
    float wave = sinf(x * 0.05f + time) * 0.5f + 0.5f;
    
    // Инвертируем и добавляем эффект
    output[output_idx] = (255 - b) * wave;      // B
    output[output_idx + 1] = (255 - g) * wave;  // G
    output[output_idx + 2] = (255 - r) * wave;  // R
    output[output_idx + 3] = a;                 // A (сохраняем прозрачность)
}
"""

def main():
    print("Инициализация захвата экрана с CUDA обработкой...")
    
    # Компиляция CUDA-ядра
    mod = SourceModule(cuda_code)
    process_frame_kernel = mod.get_function("process_frame")
    
    # Инициализация захвата экрана
    wgc_capture.set_debug(True)
    cap = wgc_capture.WGCCapture()
    
    # Переменные для измерения производительности
    start_time = time.time()
    frame_count = 0
    
    # Переменные для GPU-буфера результата
    width, height = 0, 0
    result_gpu = None
    
    print("\n--- Запуск обработки кадров на GPU с помощью CUDA ---")
    try:
        while time.time() - start_time < 20:  # Запускаем на 20 секунд
            # Захват кадра напрямую в память CUDA
            result = cap.capture_to_cuda_ptr()
            if result is None:
                time.sleep(0.001)
                continue
            
            # Распаковываем результат
            cuda_ptr, pitch, curr_width, curr_height = result
            
            # Создаем или изменяем размер выходного буфера, если размер изменился
            if width != curr_width or height != curr_height or result_gpu is None:
                width, height = curr_width, curr_height
                if result_gpu is not None:
                    result_gpu.gpudata.free()
                result_gpu = gpuarray.zeros((height, width, 4), dtype=np.uint8)
                print(f"Изменен размер буфера: {width}x{height}")
            
            # Создаем временный GPUArray для доступа к входному буферу
            input_gpu_alloc = cuda.DeviceAllocation(cuda_ptr)
            
            # Выполняем ядро для обработки кадра
            # Определяем размеры блока и сетки для CUDA
            block_size = (16, 16, 1)
            grid_size = (
                (width + block_size[0] - 1) // block_size[0],
                (height + block_size[1] - 1) // block_size[1]
            )
            
            # Передаем текущее время для анимации эффекта
            current_time = time.time() - start_time
            
            # Вызываем CUDA-ядро
            process_frame_kernel(
                input_gpu_alloc, result_gpu.gpudata,
                np.int32(width), np.int32(height), np.int32(pitch),
                np.float32(current_time),
                block=block_size, grid=grid_size
            )
            
            # Получаем результат обратно в CPU для отображения
            result_cpu = result_gpu.get()
            
            # Отображаем обработанный кадр
            cv2.imshow('WGC Screen Capture with GPU Processing', result_cpu)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
    except KeyboardInterrupt:
        print("Прервано пользователем")
    finally:
        # Очистка ресурсов
        if result_gpu is not None:
            result_gpu.gpudata.free()
        
        end_time = time.time()
        duration = end_time - start_time
        
        if frame_count > 0:
            print(f"\nОбработано {frame_count} кадров за {duration:.2f} секунд")
            print(f"Средняя частота кадров: {frame_count / duration:.1f} FPS")
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    if HAS_CUDA:
        main()
    else:
        print("Для этого примера требуется PyCUDA") 