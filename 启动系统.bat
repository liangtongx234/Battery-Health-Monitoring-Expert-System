@echo off
chcp 65001 >nul
echo ========================================
echo    Battery Health Monitoring System
echo    CBAM-CNN-Transformer with SHAP
echo ========================================
echo.

cd /d "%~dp0"

REM 尝试直接使用 streamlit
where streamlit >nul 2>&1
if %errorlevel%==0 (
    streamlit run app.py
    goto :end
)

REM 尝试 Anaconda
if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" (
    call "%USERPROFILE%\anaconda3\Scripts\activate.bat"
    streamlit run app.py
    goto :end
)

if exist "C:\ProgramData\anaconda3\Scripts\activate.bat" (
    call "C:\ProgramData\anaconda3\Scripts\activate.bat"
    streamlit run app.py
    goto :end
)

REM 尝试 python -m
python -m streamlit run app.py
if %errorlevel%==0 goto :end

echo.
echo ========================================
echo   Streamlit not found!
echo   Please install: pip install streamlit
echo ========================================

:end
pause
