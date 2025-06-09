import torch
import numpy as np
import wgc_capture
import time
import ctypes
import torch.nn.functional as F

# Включаем отладочный вывод
wgc_capture.set_debug(True)

# Инициализируем CUDA
assert torch.cuda.is_available(), "CUDA недоступен"
print(f"CUDA доступен: {torch.cuda.device_count()} устройства")
print(f"Текущее устройство: {torch.cuda.current_device()}")
print(f"Название устройства: {torch.cuda.get_device_name(0)}")

# Загружаем библиотеку D3D11
d3d11 = ctypes.WinDLL("d3d11")

# Структура для D3D11_TEXTURE2D_DESC
class D3D11_TEXTURE2D_DESC(ctypes.Structure):
    _fields_ = [
        ("Width", ctypes.c_uint),
        ("Height", ctypes.c_uint),
        ("MipLevels", ctypes.c_uint),
        ("ArraySize", ctypes.c_uint),
        ("Format", ctypes.c_uint),
        ("SampleDesc", ctypes.c_uint64),  # Упрощенно
        ("Usage", ctypes.c_uint),
        ("BindFlags", ctypes.c_uint),
        ("CPUAccessFlags", ctypes.c_uint),
        ("MiscFlags", ctypes.c_uint)
    ]

# Обертка для CUDA-PyTorch обработки текстуры D3D11
class D3D11TextureProcessor:
    def __init__(self):
        self.frame_count = 0
        self.start_time = time.time()
        
        # Подготавливаем Гауссово ядро для размытия
        kernel_size = 15
        sigma = 5.0
        
        # Создаем 1D гауссово ядро
        x = torch.arange(kernel_size) - (kernel_size - 1) / 2
        kernel_1d = torch.exp(-x**2 / (2 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        # Создаем 2D гауссово ядро
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        
        # Преобразуем в формат для свертки (C_out, C_in/groups, H, W)
        # Для каждого из 4 каналов (BGRA)
        self.kernel = kernel_2d.repeat(4, 1, 1, 1).cuda()
        
    # Старый метод с передачей numpy массива (не эффективный)
    def process_frame(self, frame, info):
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        # Проверяем, есть ли информация о текстуре
        if 'texture_ptr' not in info:
            print("Информация о текстуре недоступна")
            return
            
        print(f"Получен кадр: {frame.shape} - FPS: {fps:.2f}")
        print(f"Информация о текстуре: {info}")
        
        # Преобразуем numpy массив в тензор PyTorch и обрабатываем (неэффективно)
        tensor = torch.from_numpy(frame).cuda()
        tensor = tensor.permute(2, 0, 1).float() / 255.0
        tensor = 1.0 - tensor  # Инверсия цветов
        tensor = (tensor * 255.0).byte().permute(1, 2, 0)
        frame[:] = tensor.cpu().numpy()
        
        print(f"Кадр обработан на GPU (старый метод)")

    # Новый метод с прямым доступом к текстуре (эффективный)
    def process_texture(self, texture_info):
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        texture_ptr = texture_info['texture_ptr']
        width = texture_info['width']
        height = texture_info['height']
        d3d11_device_ptr = texture_info['d3d11_device_ptr']
        
        print(f"Получена текстура: {width}x{height} - FPS: {fps:.2f}")
        print(f"Текстура ID: {texture_ptr}, D3D11 Device: {d3d11_device_ptr}")
        
        # В реальном коде здесь был бы код для:
        # 1. Регистрации текстуры D3D11 в CUDA
        # 2. Получения указателя на память CUDA
        # 3. Использования указателя напрямую в PyTorch операциях
        # 4. Освобождения ресурсов
        
        # Примечание: для этого нужны расширения, которые не рассматриваются
        # в данном примере, такие как:
        # - cuGraphicsD3D11RegisterResource
        # - cuGraphicsMapResources
        # - cuGraphicsSubResourceGetMappedArray
        
        print(f"Текстура обработана на GPU (новый метод)")

# Создаем обработчик
processor = D3D11TextureProcessor()

# Создаем объект захвата
capture = wgc_capture.WGCCapture()

# Устанавливаем callback для обработки текстур напрямую (новый эффективный метод)
capture.set_texture_callback(processor.process_texture)

# Для демонстрации также можно установить старый callback
# capture.set_frame_callback(processor.process_frame)

try:
    print("Захват запущен. Нажмите Ctrl+C для выхода.")
    # Ждем завершения программы
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    print("Захват остановлен.") 