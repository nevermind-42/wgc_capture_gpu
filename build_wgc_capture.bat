@echo off
setlocal

echo Начинаю сборку WGC_Pybind_capture...

:: === Поиск vcvarsall.bat ===
set "VS_PATH="
for /f "usebackq tokens=*" %%i in (`dir /s /b "C:\Program Files (x86)\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvarsall.bat" 2^>nul`) do (
    set "VS_PATH=%%i"
    goto :found_vcvars
)
for /f "usebackq tokens=*" %%i in (`dir /s /b "C:\Program Files\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvarsall.bat" 2^>nul`) do (
    set "VS_PATH=%%i"
    goto :found_vcvars
)

:found_vcvars
if not defined VS_PATH (
    echo Ошибка: vcvarsall.bat не найден. Убедитесь, что Visual Studio установлена.
    goto :eof
)

echo Обнаружен vcvarsall.bat: %VS_PATH%
call "%VS_PATH%" x64
if %errorlevel% neq 0 (
    echo Ошибка при вызове vcvarsall.bat.
    goto :eof
)
echo Переменные окружения Visual Studio настроены.

:: === Переход в директорию wgc_capture ===
pushd WGC_Pybind_capture
if %errorlevel% neq 0 (
    echo Ошибка: Директория WGC_Pybind_capture не найдена.
    goto :eof
)

:: === Компиляция ===
set PYTHON_INCLUDE="C:\Users\nevermind\screen_detection\.venv312\Include"
set PYTHON_LIBS="C:\Users\nevermind\screen_detection\.venv312\Libs"
set CUDA_INCLUDE="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\include"
set CUDA_LIB="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\lib\x64"
set WIN_SDK_VERSION=10.0.26100.0
set WIN_SDK_INCLUDE="C:\Program Files (x86)\Windows Kits\10\Include\%WIN_SDK_VERSION%"
set WIN_SDK_LIB="C:\Program Files (x86)\Windows Kits\10\Lib\%WIN_SDK_VERSION%"

echo Запускаю компиляцию...
cl.exe /LD /EHsc /MD /std:c++17 /W4 /WX /await ^
/I %PYTHON_INCLUDE% ^
/I "C:\Users\nevermind\screen_detection\pybind11\include" ^
/I %CUDA_INCLUDE% ^
/I %WIN_SDK_INCLUDE%\ucrt ^
/I %WIN_SDK_INCLUDE%\shared ^
/I %WIN_SDK_INCLUDE%\um ^
/link ^
/LIBPATH:%PYTHON_LIBS% ^
/LIBPATH:%CUDA_LIB% ^
/LIBPATH:%WIN_SDK_LIB%\ucrt\x64 ^
/LIBPATH:%WIN_SDK_LIB%\um\x64 ^
wgc_capture.cpp wgc_capture_bind.cpp ^
python312.lib cudart.lib d3d11.lib dxgi.lib dcomp.lib Windows.Graphics.Capture.Interop.lib Windows.Graphics.Capture.lib Windows.Graphics.DirectX.lib Windows.Graphics.DirectX.Direct3D11.lib ^
/OUT:wgc_capture.pyd

if %errorlevel% equ 0 (
    echo Сборка успешно завершена!
) else (
    echo Ошибка при сборке. Проверьте вывод компилятора.
)

popd
endlocal
pause 