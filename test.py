import wgc_capture
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

def main():
    cap = wgc_capture.WGCCapture()
    print("Захват инициализирован.")

    # Получаем указатель на CUDA-память и pitch
    result = cap.capture_to_cuda_ptr()
    if result is None:
        print("Не удалось получить указатель на CUDA-память!")
        return

    cuda_ptr, pitch = result
    print(f"Указатель CUDA: {hex(cuda_ptr)}, pitch: {pitch}")

    # Пример: создаём PyCUDA GPUArray из этого указателя (например, BGRA 1920x1080)
    width, height = 1920, 1080  # подставь реальные значения!
    shape = (height, width, 4)
    dtype = np.uint8

    import pycuda.gpuarray as gpuarray
    # ВНИМАНИЕ: это низкоуровневый доступ, убедись, что память действительно валидна!
    frame_gpu = gpuarray.GPUArray(shape, dtype, gpudata=cuda.DeviceAllocation(cuda_ptr))

    # Теперь можно делать любые операции с frame_gpu, например, копировать на CPU:
    frame_cpu = frame_gpu.get()
    print("Кадр скопирован на CPU, shape:", frame_cpu.shape)

if __name__ == "__main__":
    main()