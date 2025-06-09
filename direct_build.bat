@echo off
setlocal

REM Загрузка конфигурации с путями
if exist build_config.bat (
    echo Загрузка конфигурации...
    call build_config.bat
) else (
    echo ОШИБКА: Файл build_config.bat не найден!
    exit /b 1
)

REM Активация MSVC компилятора
call "%VS_VCVARS%" x64

REM Показываем, какие пути используются
echo Используемые пути:
echo OpenCV include: %OPENCV_INCLUDE%
echo OpenCV lib: %OPENCV_LIB%
echo Python include: %PYTHON_INCLUDE%
echo Python lib: %PYTHON_LIBS%
echo pybind11 include: %PYBIND11_INCLUDE%

REM Создаем директорию для выходных файлов
if not exist direct_build mkdir direct_build

REM Компиляция исходных файлов
echo Компиляция исходных файлов...

cl.exe /c /EHsc /MD /std:c++17 /D_SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING ^
      /DWIN32_LEAN_AND_MEAN /O2 /DNDEBUG /await ^
      /I "%OPENCV_INCLUDE%" ^
      /I "%PYTHON_INCLUDE%" ^
      /I "%PYBIND11_INCLUDE%" ^
      wgc_capture.cpp wgc_pybind.cpp

if %ERRORLEVEL% neq 0 (
    echo ОШИБКА: Компиляция не удалась
    exit /b %ERRORLEVEL%
)

REM Линковка
echo Линковка...

link.exe /DLL /OUT:direct_build\wgc_capture.pyd ^
      wgc_capture.obj wgc_pybind.obj ^
      /LIBPATH:"%OPENCV_LIB%" %OPENCV_LIB_NAME% ^
      /LIBPATH:"%PYTHON_LIBS%" %PYTHON_LIB_NAME% ^
      windowsapp.lib d3d11.lib dxgi.lib runtimeobject.lib user32.lib shell32.lib

if %ERRORLEVEL% neq 0 (
    echo ОШИБКА: Линковка не удалась
    exit /b %ERRORLEVEL%
)

REM Копирование DLL
if exist direct_build\wgc_capture.pyd (
    echo Копирование модуля...
    copy /Y direct_build\wgc_capture.pyd .
    
    REM Если указано копирование в виртуальное окружение
    if "%COPY_TO_VENV%"=="yes" (
        if defined PYTHON_SITE_PACKAGES (
            echo Копирование модуля в виртуальное окружение Python...
            copy /Y direct_build\wgc_capture.pyd "%PYTHON_SITE_PACKAGES%\"
            
            REM Также копируем необходимую DLL OpenCV
            if defined OPENCV_BIN (
                if defined OPENCV_DLL_NAME (
                    if exist "%OPENCV_BIN%\%OPENCV_DLL_NAME%" (
                        echo Копирование OpenCV DLL в виртуальное окружение...
                        copy /Y "%OPENCV_BIN%\%OPENCV_DLL_NAME%" "%PYTHON_SITE_PACKAGES%\"
                    )
                )
            )
        )
    )
    
    echo Модуль wgc_capture.pyd успешно создан
) else (
    echo ОШИБКА: Модуль не найден после сборки
    exit /b 1
)

echo Готово!
endlocal 