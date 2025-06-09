import time
import numpy as np
import wgc_capture
import pycuda.driver as cuda
import pycuda.autoinit
# Удаляем импорты связанные с OpenGL
# from pycuda.gl import graphics_map_flags
# import pycuda.gl.autoinit
import cv2

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
    
    # Получаем размер текстуры
    width, height = cap.get_texture_size()
    print(f"Размер текстуры: {width}x{height}")
    
    # Получаем указатель на D3D11 устройство для регистрации в CUDA
    d3d11_device_ptr = cap.get_d3d11_device_ptr()
    print(f"Указатель на D3D11 устройство: {d3d11_device_ptr:x}")
    
    # PyCUDA обработка
    # В реальном приложении здесь нужно использовать CUDA для взаимодействия с D3D11
    # Это примерный код, который нужно адаптировать под конкретные нужды
    
    # Функция для обработки каждого кадра через PyCUDA
    def process_frame_gpu():
        # Получаем информацию о текстуре
        texture_info = cap.get_texture_info()
        if not texture_info:
            print("Нет доступной текстуры")
            return None
        
        print(f"Текстура: {texture_info['width']}x{texture_info['height']}, timestamp: {texture_info['timestamp']}")
        
        # В реальном приложении здесь будет код для:
        # 1. Регистрации D3D11 текстуры в CUDA (через cuda.D3D11DeviceList)
        # 2. Маппинга ресурса и получения указателя на память CUDA
        # 3. Выполнения CUDA операций с текстурой
        # 4. Размаппинга ресурса
        
        # Вместо этого в данном примере просто получаем кадр как NumPy массив
        # для демонстрации
        return cap.get_frame()
    
    # Основной цикл обработки
    print("Начинаем обработку кадров...")
    try:
        window_name = "WGC CUDA Example"
        cv2.namedWindow(window_name)
        
        while True:
            # Ждем новый кадр
            if cap.has_new_texture():
                # Обрабатываем кадр через GPU
                frame = process_frame_gpu()
                
                if frame is not None:
                    # Выводим результат
                    cv2.imshow(window_name, frame)
            
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