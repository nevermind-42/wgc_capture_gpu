import time
import numpy as np
import wgc_capture
import pycuda.driver as cuda
import pycuda.autoinit
import cv2

# CUDA ядро для инвертирования цветов
cuda_code = """
__global__ void invert_colors(unsigned char* input, unsigned char* output, int width, int height, int pitch) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * pitch + x * 4;
        
        // Читаем BGRA значения
        unsigned char b = input[idx];
        unsigned char g = input[idx + 1];
        unsigned char r = input[idx + 2];
        unsigned char a = input[idx + 3];
        
        // Инвертируем цвета
        output[idx] = 255 - b;
        output[idx + 1] = 255 - g;
        output[idx + 2] = 255 - r;
        output[idx + 3] = a;  // Альфа-канал не меняем
    }
}
"""

class CUDAProcessor:
    def __init__(self):
        # Компилируем CUDA модуль
        from pycuda.compiler import SourceModule
        self.module = SourceModule(cuda_code)
        
        # Получаем функцию из модуля
        self.invert_colors = self.module.get_function("invert_colors")
        
        # Буферы для хранения данных
        self.input_gpu = None
        self.output_gpu = None
        self.output_cpu = None
        
        # Параметры текстуры
        self.width = 0
        self.height = 0
        self.pitch = 0
        
    def process_frame(self, texture_info):
        width = texture_info['width']
        height = texture_info['height']
        pitch = texture_info['pitch']
        
        # Если размеры изменились, пересоздаем буферы
        if width != self.width or height != self.height or pitch != self.pitch:
            self._allocate_buffers(width, height, pitch)
            
        # Получаем указатель на текстуру из D3D11
        # В реальной реализации здесь должен быть код для:
        # 1. Регистрации D3D11 текстуры в CUDA
        # 2. Маппинга ресурса и получения указателя на память CUDA
        # 3. Копирования данных в input_gpu
        
        # Для примера, генерируем тестовые данные
        # Обычно этот шаг заменяется на копирование из текстуры D3D11
        test_data = np.ones((height, width, 4), dtype=np.uint8) * 128
        cuda.memcpy_htod(self.input_gpu, test_data)
        
        # Запускаем CUDA ядро
        block_size = (16, 16, 1)
        grid_size = (
            (width + block_size[0] - 1) // block_size[0],
            (height + block_size[1] - 1) // block_size[1]
        )
        
        self.invert_colors(
            self.input_gpu,
            self.output_gpu,
            np.int32(width),
            np.int32(height),
            np.int32(width * 4),  # Питч для сгенерированных данных
            block=block_size,
            grid=grid_size
        )
        
        # Копируем результат обратно в CPU
        cuda.memcpy_dtoh(self.output_cpu, self.output_gpu)
        
        # Возвращаем результат как NumPy массив
        return self.output_cpu.reshape(height, width, 4)
    
    def _allocate_buffers(self, width, height, pitch):
        print(f"Пересоздаем буферы для размера {width}x{height}")
        self.width = width
        self.height = height
        self.pitch = pitch
        
        # Освобождаем старые буферы, если они существуют
        if self.input_gpu is not None:
            self.input_gpu.free()
        if self.output_gpu is not None:
            self.output_gpu.free()
            
        # Создаем новые буферы
        buffer_size = height * width * 4  # BGRA = 4 байта на пиксель
        self.input_gpu = cuda.mem_alloc(buffer_size)
        self.output_gpu = cuda.mem_alloc(buffer_size)
        self.output_cpu = np.zeros((height, width, 4), dtype=np.uint8)
        
        print(f"Буферы успешно созданы")

def main():
    # Включаем режим отладки
    wgc_capture.set_debug(True)
    
    # Создаем экземпляр WGCCapture
    print("Создаем экземпляр WGCCapture...")
    cap = wgc_capture.WGCCapture()
    
    # Получаем информацию о мониторах
    monitors = cap.get_monitor_info()
    print(f"Найдено {len(monitors)} мониторов:")
    for i, (width, height) in enumerate(monitors):
        print(f"  Монитор {i+1}: {width}x{height}")
    
    # Проверяем наличие новой текстуры
    print("Ожидаем захват первого кадра...")
    while not cap.has_new_texture():
        time.sleep(0.1)
    
    # Создаем обработчик CUDA
    cuda_processor = CUDAProcessor()
    
    # Основной цикл обработки
    print("Начинаем обработку кадров...")
    try:
        window_name = "WGC CUDA Advanced Example"
        cv2.namedWindow(window_name)
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            # Ждем новый кадр
            if cap.has_new_texture():
                # Получаем информацию о текстуре
                texture_info = cap.get_texture_info()
                if texture_info:
                    # Обрабатываем кадр через CUDA
                    processed_frame = cuda_processor.process_frame(texture_info)
                    
                    # Конвертируем из BGRA в BGR для отображения в OpenCV
                    bgr_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGRA2BGR)
                    
                    # Выводим результат
                    cv2.imshow(window_name, bgr_frame)
                    
                    # Считаем FPS
                    frame_count += 1
                    if frame_count % 100 == 0:
                        end_time = time.time()
                        fps = frame_count / (end_time - start_time)
                        print(f"FPS: {fps:.2f}")
                        frame_count = 0
                        start_time = time.time()
            
            # Проверяем нажатие клавиши
            key = cv2.waitKey(1)
            if key == ord('q') or key == 27:  # q или Esc
                break
                
    except KeyboardInterrupt:
        print("Прервано пользователем")
    finally:
        cv2.destroyAllWindows()
        print("Завершение работы...")

if __name__ == "__main__":
    main() 