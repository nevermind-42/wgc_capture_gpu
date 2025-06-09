cmake_minimum_required(VERSION 3.18)
project(wgc_capture)

# Находим CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Находим Python
find_package(Python COMPONENTS Interpreter Development REQUIRED)
include_directories(${Python_INCLUDE_DIRS})

# Находим pybind11
# find_package(pybind11 CONFIG REQUIRED)

# Добавляем исходные файлы
add_library(wgc_capture SHARED
    wgc_capture.cpp
    wgc_capture_bind.cpp
)

# Линкуем с CUDA
target_link_libraries(wgc_capture PRIVATE
    pybind11::module
    opencv_world410
    d3d11
    dxgi
    dcomp
    windowsapp
    ${CUDA_LIBRARIES}
)

# Устанавливаем флаги компиляции
target_compile_features(wgc_capture PRIVATE cxx_std_17)
target_compile_options(wgc_capture PRIVATE /W4 /WX-)

# Устанавливаем имя выходного файла
set_target_properties(wgc_capture PROPERTIES
    PREFIX ""
    SUFFIX ".pyd"
)

# Вместо find_package(pybind11 CONFIG REQUIRED)
add_subdirectory(${CMAKE_SOURCE_DIR}/../pybind11 pybind11_build)
# find_package(pybind11 CONFIG REQUIRED)

include_directories("C:/Tools/opencv/4.1.0/opencv/build/include")

target_link_directories(wgc_capture PRIVATE "C:/Tools/opencv/4.1.0/opencv/build/x64/vc16/lib")
target_link_libraries(wgc_capture PRIVATE
    pybind11::module
    opencv_world4100
    d3d11
    dxgi
    dcomp
    windowsapp
    ${CUDA_LIBRARIES}
) 