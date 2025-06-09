import wgc_capture
import time
import cv2
import numpy as np
import os

def save_frame(frame_data, pitch, width, height, timestamp):
    """Callback функция для сохранения кадра"""
    # Создаем numpy массив из данных кадра
    frame = np.frombuffer(frame_data, dtype=np.uint8)
    frame = frame.reshape((height, width, 4))  # BGRA формат
    
    # Конвертируем из BGRA в BGR
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    
    # Сохраняем кадр
    filename = f"frame_{timestamp}.png"
    cv2.imwrite(filename, frame_bgr)
    print(f"Сохранен кадр: {filename}")

def main():
    try:
        print("Тестирование WGC модуля в Python...")
        
        # Создаем экземпляр захвата
        capture = wgc_capture.WGCCapture()
        
        # Получаем информацию о мониторах
        monitor_info = capture.get_monitor_info()
        print(f"Найдено мониторов: {len(monitor_info)}")
        for i, (left, top, right, bottom) in enumerate(monitor_info):
            print(f"Монитор {i}: ({left},{top}) - ({right},{bottom})")
        
        # Устанавливаем callback для сохранения кадров
        capture.set_frame_callback(save_frame)
        
        # Запускаем захват
        print("Запуск захвата...")
        if not capture.start_capture():
            print("Ошибка при запуске захвата!")
            return
        
        # Ждем 2 секунды
        print("Захват запущен, ожидание 2 секунды...")
        time.sleep(2)
        
        # Приостанавливаем захват
        print("Приостановка захвата...")
        if not capture.pause_capture():
            print("Ошибка при приостановке захвата!")
            return
        
        # Ждем 1 секунду
        print("Захват приостановлен, ожидание 1 секунда...")
        time.sleep(1)
        
        # Возобновляем захват
        print("Возобновление захвата...")
        if not capture.resume_capture():
            print("Ошибка при возобновлении захвата!")
            return
        
        # Ждем еще 2 секунды
        print("Захват возобновлен, ожидание 2 секунды...")
        time.sleep(2)
        
        # Тестируем прямой захват в CUDA
        print("Тестирование прямого захвата в CUDA...")
        cuda_ptr, pitch, width, height, timestamp = capture.capture_to_cuda()
        
        if cuda_ptr:
            print(f"Успешный захват в CUDA: {width}x{height} (pitch: {pitch})")
            
            # Создаем numpy массив из данных кадра
            frame = np.frombuffer(cuda_ptr, dtype=np.uint8)
            frame = frame.reshape((height, width, 4))  # BGRA формат
            
            # Конвертируем из BGRA в BGR
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            # Сохраняем кадр
            filename = f"frame_cuda_{timestamp}.png"
            cv2.imwrite(filename, frame_bgr)
            print(f"Сохранен CUDA кадр: {filename}")
        else:
            print("Ошибка при захвате в CUDA!")
        
        # Останавливаем захват
        print("Остановка захвата...")
        capture.stop_capture()
        
        print("Тест завершен успешно!")
        
    except Exception as e:
        print(f"Ошибка: {str(e)}")

if __name__ == "__main__":
    main() 