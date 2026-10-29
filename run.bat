@echo off
chcp 65001 >nul
rem Запуск All the girls без консольного окна (журнал пишется в logs\app.log).
rem Для запуска с консолью: .venv_main\Scripts\python.exe app.py
cd /d "%~dp0"
if not exist ".venv_main\Scripts\pythonw.exe" (
    echo Не найдено окружение .venv_main. Создайте его и установите зависимости:
    echo   py -3.9 -m venv .venv_main
    echo   .venv_main\Scripts\python.exe -m pip install -r requirements-main.txt
    pause
    exit /b 1
)
start "" ".venv_main\Scripts\pythonw.exe" app.py %*
