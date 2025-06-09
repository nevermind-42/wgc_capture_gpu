@echo off
REM ========================================================================
REM Файл конфигурации с путями для сборки модуля WGC Capture
REM ========================================================================

REM Пути к Visual Studio
set VS_PATH=C:\Program Files\Microsoft Visual Studio\2022\Community
set VS_VCVARS=%VS_PATH%\VC\Auxiliary\Build\vcvarsall.bat

REM Пути к OpenCV
set OPENCV_DIR=C:\Tools\opencv\4.1.0\build
set OPENCV_INCLUDE=%OPENCV_DIR%\include
set OPENCV_LIB=%OPENCV_DIR%\x64\vc16\lib
set OPENCV_BIN=%OPENCV_DIR%\x64\vc16\bin
set OPENCV_LIB_NAME=opencv_world4100.lib
set OPENCV_DLL_NAME=opencv_world4100.dll

REM Пути к Python (базовая установка для компиляции)
set PYTHON_BASE=C:\Users\nevermind\AppData\Local\Programs\Python\Python312
set PYTHON_INCLUDE=%PYTHON_BASE%\Include
set PYTHON_LIBS=%PYTHON_BASE%\libs
set PYTHON_LIB_NAME=python312.lib

REM Пути к виртуальному окружению Python
set PYTHON_VENV=C:\Users\nevermind\screen_detection\.venv312
set PYTHON_EXE=%PYTHON_VENV%\Scripts\python.exe
set PYTHON_SITE_PACKAGES=%PYTHON_VENV%\Lib\site-packages

REM После компиляции копировать .pyd файл в директорию виртуального окружения
set COPY_TO_VENV=yes

REM Пути к pybind11
set PYBIND11_DIR=..\pybind11
set PYBIND11_INCLUDE=%PYBIND11_DIR%\include

REM Дополнительные библиотеки для линковки
set ADDITIONAL_LIBS=windowsapp.lib d3d11.lib dxgi.lib runtimeobject.lib user32.lib shell32.lib

REM Флаги компилятора
set CL_FLAGS=/std:c++17 /EHsc /MD /O2 /DNDEBUG /LD /D_SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING /DWIN32_LEAN_AND_MEAN

REM Имя выходного файла
set OUTPUT_NAME=wgc_capture.pyd

echo Конфигурация загружена.
echo OpenCV: %OPENCV_DIR%
echo Python (базовый): %PYTHON_BASE%
echo Python (виртуальное окружение): %PYTHON_VENV%
echo pybind11: %PYBIND11_DIR% 