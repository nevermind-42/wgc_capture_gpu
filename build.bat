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

echo Создание директории сборки...
if not exist build mkdir build
cd build

echo Настройка CMake...
cmake .. -A x64 -DCMAKE_BUILD_TYPE=Release

echo Сборка проекта...
cmake --build . --config Release

echo Копирование собранного модуля...
if exist "Release\wgc_pybind*.pyd" (
    echo Копирование модуля из Release...
    
    REM Явно переименовываем модуль в wgc_capture.pyd
    REM Копируем в текущий каталог проекта
    copy /Y "Release\wgc_pybind*.pyd" "..\wgc_capture.pyd"
    
    REM Если указано копирование в виртуальное окружение
    if "%COPY_TO_VENV%"=="yes" (
        if defined PYTHON_SITE_PACKAGES (
            echo Копирование модуля в виртуальное окружение Python...
            copy /Y "Release\wgc_pybind*.pyd" "%PYTHON_SITE_PACKAGES%\wgc_capture.pyd"
            
            REM Также копируем необходимую DLL OpenCV, если она существует
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
    
    echo Модуль wgc_capture.pyd успешно скопирован
) else (
    echo ОШИБКА: Не удалось найти модуль wgc_pybind*.pyd в папке Release
    exit /b 1
)

cd ..

echo Готово!

endlocal 
