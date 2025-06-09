import wgc_capture
import numpy as np
import cv2
import time

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    import pycuda.gpuarray as gpuarray
    HAS_CUDA = True
except ImportError:
    print("PyCUDA не установлен, будет использован альтернативный режим")
    HAS_CUDA = False

def main():
    print("Инициализация захвата экрана...")
    # Включаем отладочный вывод
    wgc_capture.set_debug(True)
    cap = wgc_capture.WGCCapture()
    
    # Выполняем 10 секунд захвата кадров
    start_time = time.time()
    frame_count = 0
    
    print("\n--- Демонстрация захвата напрямую в CUDA память ---")
    if HAS_CUDA:
        while time.time() - start_time < 10:
            # Захват кадра напрямую в память CUDA
            result = cap.capture_to_cuda_ptr()
            if result is None:
                time.sleep(0.001)  # Небольшая задержка, если кадр недоступен
                continue
            
            # Распаковываем результат (указатель на память CUDA, pitch, width, height)
            cuda_ptr, pitch, width, height = result
            
            # Создаем GPUArray из указателя (для дальнейшей обработки на GPU)
            gpu_alloc = cuda.DeviceAllocation(cuda_ptr)
            
            # ВАЖНО: мы не владеем этой памятью, она принадлежит WGC
            # Поэтому мы не должны её освобождать
            
            # Создаем GPUArray для обработки/отображения
            # Для демонстрации, просто копируем данные на CPU
            frame_gpu = gpuarray.GPUArray((height, width, 4), np.uint8, gpudata=gpu_alloc)
            frame_cpu = frame_gpu.get()
            
            # Обработка кадра (например, преобразование BGR в RGB)
            frame_rgb = cv2.cvtColor(frame_cpu, cv2.COLOR_BGRA2RGB)
            
            # Отображаем кадр (можно закомментировать для бенчмаркинга)
            cv2.imshow('WGC Screen Capture (GPU Mode)', frame_rgb)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
    else:
        print("PyCUDA не установлен, используем обычный режим")
        while time.time() - start_time < 10:
            # Получаем кадр вместе с метаданными
            frame, info = cap.get_frame_with_info()
            
            if frame.size == 0:
                time.sleep(0.001)  # Небольшая задержка, если кадр недоступен
                continue
            
            # Выводим метаданные для первого кадра
            if frame_count == 0:
                print(f"Метаданные кадра: {info}")
            
            # Отображаем кадр (можно закомментировать для бенчмаркинга)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
            cv2.imshow('WGC Screen Capture (CPU Mode)', frame_rgb)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\nЗахвачено {frame_count} кадров за {duration:.2f} секунд")
    print(f"Средняя частота кадров: {frame_count / duration:.1f} FPS")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 