#include <cuda_runtime.h>

// CUDA kernel для конвертации BGRA -> RGB
__global__ void bgra_to_rgb_kernel(
    const unsigned char* bgra,
    unsigned char* rgb,
    int width,
    int height,
    int bgra_pitch,
    int rgb_pitch
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        // Получаем указатели на текущую строку
        const unsigned char* bgra_row = bgra + y * bgra_pitch;
        unsigned char* rgb_row = rgb + y * rgb_pitch;
        
        // Конвертируем BGRA -> RGB
        rgb_row[x * 3 + 0] = bgra_row[x * 4 + 2]; // R
        rgb_row[x * 3 + 1] = bgra_row[x * 4 + 1]; // G
        rgb_row[x * 3 + 2] = bgra_row[x * 4 + 0]; // B
    }
}

// Функция для запуска kernel
extern "C" void convert_bgra_to_rgb(
    const unsigned char* bgra,
    unsigned char* rgb,
    int width,
    int height,
    int bgra_pitch,
    int rgb_pitch,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    bgra_to_rgb_kernel<<<grid, block, 0, stream>>>(
        bgra, rgb, width, height, bgra_pitch, rgb_pitch
    );
} 