@echo off
echo ===========================================
echo    SliTraNet Environment Setup
echo ===========================================
echo.

echo Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo Error: Failed to create virtual environment
    echo Please make sure Python 3.x is installed
    pause
    exit /b 1
)

echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo Error: Failed to activate virtual environment
    pause
    exit /b 1
)

echo.
echo Installing packages...
pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo Error: Failed to install packages
    pause
    exit /b 1
)

echo.
echo ===========================================
echo    Setup Complete!
echo ===========================================
echo.
echo Usage:
echo   1. Prepare video file
echo   2. Run run_inference.bat
echo   3. Pass video file path as argument
echo.
echo Example: run_inference.bat "C:\path\to\video.mp4"
echo.
pause