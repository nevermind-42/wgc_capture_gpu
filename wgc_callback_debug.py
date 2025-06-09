import numpy as np
import time
import wgc_capture
import sys
import ctypes

# Включаем отладочный вывод
wgc_capture.set_debug(True)
print("Инициализация модуля захвата с включенной отладкой")

# Счетчик кадров и время начала
frame_count = 0
start_time = None

# Функция обратного вызова для отладки
def debug_callback(frame, info):
    global frame_count, start_time
    
    # При первом кадре инициализируем время
    if start_time is None:
        start_time = time.time()
    
    # Детальный вывод информации о кадре
    print(f"\n---- Обратный вызов {frame_count+1} ----")
    print(f"Время: {time.time():.2f}")
    print(f"Тип frame: {type(frame)}")
    
    if frame is None:
        print("ОШИБКА: frame is None")
    elif not isinstance(frame, np.ndarray):
        print(f"ОШИБКА: frame не является numpy.ndarray, а {type(frame)}")
    else:
        print(f"Размер frame: {frame.shape}")
        print(f"Тип данных: {frame.dtype}")
        print(f"Min/Max значения: {frame.min()}/{frame.max()}")
    
    # Выводим информацию о полученных метаданных
    print("Информация:")
    if info:
        for key, value in info.items():
            print(f"  {key}: {value}")
    else:
        print("  Нет метаданных")
    
    # Если кадр получен успешно, увеличиваем счетчик
    if frame is not None and isinstance(frame, np.ndarray) and frame.size > 0:
        frame_count += 1
        
        # Инвертируем цвета для визуального эффекта
        if frame.ndim == 3:  # Проверяем, что это многоканальное изображение
            np.subtract(255, frame, out=frame)
        
        # Выводим FPS каждые 10 кадров
        if frame_count % 10 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            print(f"Обработано {frame_count} кадров, FPS: {fps:.2f}")

# Функция обратного вызова для обычного колбэка (старый метод)
def frame_callback(frame, info):
    print(f"Frame callback: Получен кадр размером {frame.shape}")
    print(f"Frame callback: Метаданные: {info}")

# Функция обратного вызова для текстуры (новый метод)
def texture_callback(texture_info):
    print(f"Texture callback: Получены данные текстуры:")
    print(f"  - Указатель на текстуру: {texture_info['texture_ptr']}")
    print(f"  - Размер: {texture_info['width']}x{texture_info['height']}")
    print(f"  - Timestamp: {texture_info['timestamp']}")
    print(f"  - Указатель на D3D11 устройство: {texture_info['d3d11_device_ptr']}")

try:
    # Инициализация захвата
    print("Инициализация захвата...")
    capture = wgc_capture.WGCCapture()
    print("Захват инициализирован")
    
    # Сначала тестируем старый метод колбэка
    print("\n==== Тестирование стандартного колбэка с копированием кадра ====")
    capture.set_frame_callback(frame_callback)
    time.sleep(2)  # Ждем несколько кадров

    # Теперь тестируем новый метод колбэка с прямой передачей текстуры
    print("\n==== Тестирование нового колбэка с прямой передачей текстуры ====")
    capture.set_texture_callback(texture_callback)
    time.sleep(2)  # Ждем несколько кадров

    print("\nТест завершен. Проверьте логи на наличие ошибок.")

except Exception as e:
    print(f"Ошибка: {e}")
    import traceback
    traceback.print_exc()
    
print("Тест завершен") 