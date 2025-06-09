cmake_minimum_required(VERSION 3.18)
project(wgc_capture)

# Находим CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Находим Python
find_package(Python COMPONENTS Interpreter Development REQUIRED)
include_directories(${Python_INCLUDE_DIRS})

# Добавляем исходные файлы для Python модуля
add_library(wgc_capture SHARED
    wgc_capture.cpp
    wgc_pybind.cpp
)

# Добавляем статическую библиотеку для тестов
add_library(wgc_capture_static STATIC
    wgc_capture.cpp
)

# Добавляем тестовый исполняемый файл
add_executable(test_capture
    test_capture.cpp
)

# Вместо find_package(pybind11 CONFIG REQUIRED)
add_subdirectory(${CMAKE_SOURCE_DIR}/../pybind11 pybind11_build)

include_directories("C:/Tools/opencv/4.1.0/opencv/build/include")

target_link_directories(wgc_capture PRIVATE "C:/Tools/opencv/4.1.0/opencv/build/x64/vc16/lib")
target_link_directories(wgc_capture_static PRIVATE "C:/Tools/opencv/4.1.0/opencv/build/x64/vc16/lib")
target_link_directories(test_capture PRIVATE "C:/Tools/opencv/4.1.0/opencv/build/x64/vc16/lib")

# Линкуем с библиотеками
target_link_libraries(wgc_capture PRIVATE
    pybind11::module
    opencv_world410
    d3d11
    dxgi
    dcomp
    windowsapp
    ${CUDA_LIBRARIES}
)

target_link_libraries(wgc_capture_static PRIVATE
    opencv_world410
    d3d11
    dxgi
    dcomp
    windowsapp
    ${CUDA_LIBRARIES}
)

target_link_libraries(test_capture PRIVATE
    wgc_capture_static
    opencv_world410
    d3d11
    dxgi
    dcomp
    windowsapp
    ${CUDA_LIBRARIES}
)

# Устанавливаем флаги компиляции
target_compile_features(wgc_capture PRIVATE cxx_std_17)
target_compile_features(wgc_capture_static PRIVATE cxx_std_17)
target_compile_features(test_capture PRIVATE cxx_std_17)
target_compile_options(wgc_capture PRIVATE /W4 /WX-)
target_compile_options(wgc_capture_static PRIVATE /W4 /WX-)
target_compile_options(test_capture PRIVATE /W4 /WX-)

# Устанавливаем имя выходного файла
set_target_properties(wgc_capture PROPERTIES
    PREFIX ""
    SUFFIX ".pyd"
)