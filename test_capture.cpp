#include "wgc_capture.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <opencv2/opencv.hpp>

// Функция для сохранения кадра в файл
void save_frame_to_file(void* cuda_ptr, size_t pitch, int width, int height, int64_t timestamp) {
    // Создаем CPU буфер для копирования данных
    std::vector<uint8_t> cpu_buffer(width * height * 4);
    
    // Копируем данные из CUDA в CPU
    cudaMemcpy2D(cpu_buffer.data(), width * 4,
                 cuda_ptr, pitch,
                 width * 4, height,
                 cudaMemcpyDeviceToHost);
    
    // Создаем OpenCV матрицу
    cv::Mat frame(height, width, CV_8UC4, cpu_buffer.data());
    
    // Конвертируем из BGRA в BGR
    cv::Mat bgr_frame;
    cv::cvtColor(frame, bgr_frame, cv::COLOR_BGRA2BGR);
    
    // Сохраняем кадр
    std::string filename = "frame_" + std::to_string(timestamp) + ".png";
    cv::imwrite(filename, bgr_frame);
    std::cout << "Сохранен кадр: " << filename << std::endl;
}

int main() {
    try {
        std::cout << "Тестирование WGCCapture..." << std::endl;
        
        // Создаем экземпляр захвата
        WGCCapture capture;
        
        // Получаем информацию о мониторах
        auto monitor_info = capture.get_monitor_info();
        std::cout << "Найдено мониторов: " << monitor_info.size() << std::endl;
        for (size_t i = 0; i < monitor_info.size(); ++i) {
            auto [left, top, right, bottom] = monitor_info[i];
            std::cout << "Монитор " << i << ": " 
                      << "(" << left << "," << top << ") - "
                      << "(" << right << "," << bottom << ")" << std::endl;
        }
        
        // Устанавливаем callback для сохранения кадров
        capture.set_frame_callback(save_frame_to_file);
        
        // Запускаем захват
        std::cout << "Запуск захвата..." << std::endl;
        if (!capture.start_capture()) {
            std::cerr << "Ошибка при запуске захвата!" << std::endl;
            return 1;
        }
        
        // Ждем 2 секунды
        std::cout << "Захват запущен, ожидание 2 секунды..." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(2));
        
        // Приостанавливаем захват
        std::cout << "Приостановка захвата..." << std::endl;
        if (!capture.pause_capture()) {
            std::cerr << "Ошибка при приостановке захвата!" << std::endl;
            return 1;
        }
        
        // Ждем 1 секунду
        std::cout << "Захват приостановлен, ожидание 1 секунда..." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(1));
        
        // Возобновляем захват
        std::cout << "Возобновление захвата..." << std::endl;
        if (!capture.resume_capture()) {
            std::cerr << "Ошибка при возобновлении захвата!" << std::endl;
            return 1;
        }
        
        // Ждем еще 2 секунды
        std::cout << "Захват возобновлен, ожидание 2 секунды..." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(2));
        
        // Тестируем прямой захват в CUDA
        std::cout << "Тестирование прямого захвата в CUDA..." << std::endl;
        void* cuda_ptr = nullptr;
        size_t pitch = 0;
        int width = 0, height = 0;
        int64_t timestamp = 0;
        
        if (capture.capture_to_cuda(&cuda_ptr, &pitch, &width, &height, &timestamp)) {
            std::cout << "Успешный захват в CUDA: " 
                      << width << "x" << height 
                      << " (pitch: " << pitch << ")" << std::endl;
            
            // Сохраняем кадр
            save_frame_to_file(cuda_ptr, pitch, width, height, timestamp);
        } else {
            std::cerr << "Ошибка при захвате в CUDA!" << std::endl;
        }
        
        // Останавливаем захват
        std::cout << "Остановка захвата..." << std::endl;
        capture.stop_capture();
        
        std::cout << "Тест завершен успешно!" << std::endl;
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Ошибка: " << e.what() << std::endl;
        return 1;
    }
} 