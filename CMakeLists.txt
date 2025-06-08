cmake_minimum_required(VERSION 3.15)
project(wgc_pybind)

# Устанавливаем C++17 стандарт
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Добавляем пути к OpenCV из переменных окружения
if(DEFINED ENV{OPENCV_INCLUDE})
    message(STATUS "Using OpenCV include: $ENV{OPENCV_INCLUDE}")
    include_directories($ENV{OPENCV_INCLUDE})
else()
    message(FATAL_ERROR "OPENCV_INCLUDE not set. Run build_config.bat first")
endif()

if(DEFINED ENV{OPENCV_LIB})
    message(STATUS "Using OpenCV lib: $ENV{OPENCV_LIB}")
    link_directories($ENV{OPENCV_LIB})
else()
    message(FATAL_ERROR "OPENCV_LIB not set. Run build_config.bat first")
endif()

# Настраиваем путь к pybind11
# (предполагается, что pybind11 находится в корне проекта)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../pybind11 ${CMAKE_CURRENT_BINARY_DIR}/pybind11)

# Настройки для Windows
if(WIN32)
    # Для WinRT
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /await")
    
    # Включаем требуемые библиотеки
    set(REQUIRED_LIBS 
        windowsapp
        d3d11
        dxgi
        dwmapi
        shcore
    )
endif()

# Исходные файлы
set(SOURCES 
    wgc_capture.cpp
    wgc_pybind.cpp
)

# Создание модуля Python
pybind11_add_module(wgc_pybind ${SOURCES})

# Устанавливаем имя выходного файла, если указано
if(DEFINED ENV{OUTPUT_NAME})
    message(STATUS "Setting output name: $ENV{OUTPUT_NAME}")
    set_target_properties(wgc_pybind PROPERTIES OUTPUT_NAME "$ENV{OUTPUT_NAME}")
endif()

# Линкуем необходимые библиотеки
target_link_libraries(wgc_pybind PRIVATE ${REQUIRED_LIBS})

# Добавляем библиотеку OpenCV
if(DEFINED ENV{OPENCV_LIB_NAME})
    message(STATUS "Using OpenCV lib name: $ENV{OPENCV_LIB_NAME}")
    target_link_libraries(wgc_pybind PRIVATE $ENV{OPENCV_LIB_NAME})
else()
    message(FATAL_ERROR "OPENCV_LIB_NAME not set. Run build_config.bat first")
endif()

# Цель установки для Python
install(TARGETS wgc_pybind DESTINATION .) 