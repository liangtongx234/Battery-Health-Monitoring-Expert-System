@echo off
chcp 65001 >nul
echo.

cd /d "%~dp0"

REM 检查data目录
if not exist "data" (
    echo Error: data directory not found!
    echo Please create 'data' folder and put your CSV files.
    pause
    exit /b
)

echo Data directory found.
echo Starting batch training...
echo.

REM 尝试运行Python脚本
python train_all_models.py

if %errorlevel% neq 0 (
    echo.
    echo   Training failed!
    echo   Please check error messages above.
)

echo.
pause
