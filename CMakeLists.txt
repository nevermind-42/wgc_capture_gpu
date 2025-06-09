cmake_minimum_required(VERSION 3.15)
project(wgc_capture)

# Стандарт C++17 (необходим для WinRT)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Находим CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Находим Python
find_package(Python COMPONENTS Interpreter Development REQUIRED)
include_directories(${Python_INCLUDE_DIRS})

# Пути к OpenCV
set(OPENCV_INCLUDE "C:/Tools/opencv/4.1.0/opencv/build/include")
set(OPENCV_LIB "C:/Tools/opencv/4.1.0/opencv/build/x64/vc16/lib")
set(OPENCV_LIB_NAME "opencv_world4100")
include_directories(${OPENCV_INCLUDE})
link_directories(${OPENCV_LIB})

# Настройки для Windows и WinRT
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /await")
set(REQUIRED_LIBS 
    windowsapp
    d3d11
    dxgi
    dwmapi
    shcore
)

# Добавляем путь к pybind11
add_subdirectory(${CMAKE_SOURCE_DIR}/../pybind11 pybind11_build)

# Исходные файлы для модуля
add_library(wgc_capture SHARED
    wgc_capture.cpp
    wgc_pybind.cpp
)

# Линкуем с необходимыми библиотеками
target_link_libraries(wgc_capture PRIVATE
    pybind11::module
    ${OPENCV_LIB_NAME}
    ${REQUIRED_LIBS}
    ${CUDA_LIBRARIES}
)

# Копирование CUDA DLL в директорию сборки
add_custom_command(TARGET wgc_capture POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CUDA_TOOLKIT_ROOT_DIR}/bin/cudart64_12.dll"
    $<TARGET_FILE_DIR:wgc_capture>
)

# Устанавливаем имя выходного файла (.pyd для Python)
set_target_properties(wgc_capture PROPERTIES
    PREFIX ""
    SUFFIX ".pyd"
) 