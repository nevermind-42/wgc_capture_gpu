import time
import numpy as np
import matplotlib.pyplot as plt
import wgc_capture
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import ctypes

# Количество кадров для теста
TOTAL_FRAMES = 300

# CUDA ядро для инвертирования цветов
cuda_kernel = """
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
        self.mod = SourceModule(cuda_kernel)
        self.invert_colors = self.mod.get_function("invert_colors")
        
        # Буферы для обработки
        self.d_input = None
        self.d_output = None
        self.h_output = None
        
        # Размеры
        self.width = 0
        self.height = 0
        self.pitch = 0
        
    def initialize(self, width, height):
        """Инициализация ресурсов CUDA"""
        try:
            self.width = width
            self.height = height
            self.pitch = width * 4  # BGRA = 4 байта на пиксель
            
            print(f"Инициализация CUDA для изображения {width}x{height}")
            
            # Выделяем память на GPU для входных и выходных данных
            buffer_size = self.height * self.pitch
            self.d_input = cuda.mem_alloc(buffer_size)
            self.d_output = cuda.mem_alloc(buffer_size)
            
            # Выделяем память на CPU для выходных данных
            self.h_output = np.zeros((height, width, 4), dtype=np.uint8)
            
            print(f"Выделено памяти: {buffer_size / (1024*1024):.2f} МБ на GPU")
            
            return True
        except Exception as e:
            print(f"Ошибка при инициализации CUDA: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def process_frame(self, frame_data):
        """Обработка кадра на GPU с копированием CPU->GPU->CPU"""
        try:
            if self.d_input is None or self.d_output is None:
                return None
                
            # Копируем данные с CPU на GPU
            cuda.memcpy_htod(self.d_input, frame_data)
            
            # Задаем размеры блоков и сетки для ядра
            block_size = (16, 16, 1)
            grid_size = (
                (self.width + block_size[0] - 1) // block_size[0],
                (self.height + block_size[1] - 1) // block_size[1]
            )
            
            # Вызываем CUDA ядро
            self.invert_colors(
                self.d_input,
                self.d_output,
                np.int32(self.width),
                np.int32(self.height),
                np.int32(self.pitch),
                block=block_size,
                grid=grid_size
            )
            
            # Копируем результат с GPU на CPU
            cuda.memcpy_dtoh(self.h_output, self.d_output)
            
            return self.h_output
        except Exception as e:
            print(f"Ошибка при обработке кадра на GPU: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def cleanup(self):
        """Освобождение ресурсов"""
        try:
            if self.d_input:
                self.d_input.free()
            if self.d_output:
                self.d_output.free()
            print("Ресурсы CUDA освобождены")
        except Exception as e:
            print(f"Ошибка при освобождении ресурсов: {e}")


def main():
    # Включаем режим отладки
    wgc_capture.set_debug(True)
    print(f"Тест CUDA обработки для {TOTAL_FRAMES} кадров (с копированием CPU<->GPU)")
    
    # Создаем экземпляр WGCCapture
    print("Инициализация WGCCapture...")
    cap = wgc_capture.WGCCapture()
    
    # Ожидаем первый кадр
    print("Ожидание первого кадра...")
    while not cap.has_new_texture():
        time.sleep(0.1)
    
    # Получаем размер текстуры
    width, height = cap.get_texture_size()
    print(f"Размер кадра: {width}x{height}")
    
    # Создаем процессор CUDA
    cuda_processor = CUDAProcessor()
    
    # Инициализируем CUDA для обработки кадров указанного размера
    if not cuda_processor.initialize(width, height):
        print("Не удалось инициализировать CUDA")
        return
    
    # Массивы для хранения данных производительности
    copy_to_gpu_times = []   # Время копирования CPU->GPU
    process_times = []       # Время выполнения CUDA ядра
    copy_from_gpu_times = [] # Время копирования GPU->CPU
    total_times = []         # Общее время обработки
    fps_values = []          # FPS для каждых 10 кадров
    
    # Получаем кадры и измеряем производительность
    processed_frames = 0
    total_start_time = time.time()
    
    print(f"Начало теста производительности. Сбор данных для {TOTAL_FRAMES} кадров...")
    
    try:
        while processed_frames < TOTAL_FRAMES:
            # Получаем кадр из WGCCapture
            frame = cap.get_frame()
            
            if frame is not None and frame.size > 0:
                # Измеряем время обработки
                frame_start_time = time.time()
                
                # Копируем данные на GPU, измеряем время
                copy_to_gpu_start = time.time()
                # Обрабатываем кадр на GPU
                result = cuda_processor.process_frame(frame)
                frame_end_time = time.time()
                
                if result is not None:
                    # Вычисляем и сохраняем время обработки
                    total_time_ms = (frame_end_time - frame_start_time) * 1000
                    total_times.append(total_time_ms)
                    
                    # Каждые 10 кадров вычисляем FPS
                    if processed_frames % 10 == 0 and processed_frames > 0:
                        current_time = time.time()
                        segment_time = current_time - total_start_time
                        current_fps = processed_frames / segment_time
                        fps_values.append(current_fps)
                        
                        # Выводим текущую статистику
                        print(f"Кадр {processed_frames}/{TOTAL_FRAMES}, FPS: {current_fps:.2f}, Время: {total_time_ms:.2f} мс")
                    
                    processed_frames += 1
            
            # Небольшая пауза, чтобы не перегружать систему
            time.sleep(0.001)
    except KeyboardInterrupt:
        print("Тест прерван пользователем")
    finally:
        # Освобождаем ресурсы
        cuda_processor.cleanup()
    
    # Вычисляем итоговую статистику
    total_time = time.time() - total_start_time
    average_fps = processed_frames / total_time
    average_process_time = np.mean(total_times)
    min_process_time = np.min(total_times)
    max_process_time = np.max(total_times)
    p95_process_time = np.percentile(total_times, 95)
    p99_process_time = np.percentile(total_times, 99)
    
    # Выводим общую статистику
    print("\n=== Результаты теста CUDA обработки ===")
    print(f"Всего кадров: {processed_frames}")
    print(f"Общее время: {total_time:.2f} секунд")
    print(f"Средний FPS: {average_fps:.2f}")
    print(f"Среднее время обработки: {average_process_time:.2f} мс")
    print(f"Минимальное время обработки: {min_process_time:.2f} мс")
    print(f"Максимальное время обработки: {max_process_time:.2f} мс")
    print(f"95-й процентиль времени обработки: {p95_process_time:.2f} мс")
    print(f"99-й процентиль времени обработки: {p99_process_time:.2f} мс")
    
    # Строим графики
    plt.figure(figsize=(15, 10))
    
    # График времени обработки кадров
    plt.subplot(2, 1, 1)
    plt.plot(range(len(total_times)), total_times, 'b-')
    plt.axhline(y=average_process_time, color='r', linestyle='--', label=f'Среднее: {average_process_time:.2f} мс')
    plt.axhline(y=p95_process_time, color='g', linestyle='--', label=f'95-й процентиль: {p95_process_time:.2f} мс')
    plt.title('Время обработки кадра на GPU (с копированием)')
    plt.xlabel('Номер кадра')
    plt.ylabel('Время (мс)')
    plt.grid(True)
    plt.legend()
    
    # График FPS
    frame_indices = range(10, processed_frames + 1, 10)
    if len(fps_values) > 0:  # Проверяем, что есть данные для графика
        plt.subplot(2, 1, 2)
        plt.plot(frame_indices[:len(fps_values)], fps_values, 'g-')
        plt.axhline(y=average_fps, color='r', linestyle='--', label=f'Средний FPS: {average_fps:.2f}')
        plt.title('FPS во время теста (CUDA с копированием)')
        plt.xlabel('Номер кадра')
        plt.ylabel('FPS')
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('gpu_cuda_copy_results.png')
    plt.show()
    
    # Также сохраняем данные в CSV для дальнейшего анализа
    with open('gpu_cuda_copy_data.csv', 'w') as f:
        f.write('frame,total_time_ms\n')
        for i in range(len(total_times)):
            f.write(f'{i+1},{total_times[i]:.2f}\n')
    
    print("\nГрафики сохранены в файл 'gpu_cuda_copy_results.png'")
    print("Данные сохранены в файл 'gpu_cuda_copy_data.csv'")

if __name__ == "__main__":
    main() 