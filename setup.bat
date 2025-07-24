@echo off
echo ===========================================
echo    SliTraNet Environment Setup (Integrated)
echo    image_dupp functionality integrated
echo ===========================================
echo.

echo Creating virtual environment...
if exist venv (
    echo Removing existing virtual environment...
    rmdir /s /q venv
)
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
echo Installing packages from requirements.txt...
pip install --upgrade pip
if not exist requirements.txt (
    echo Error: requirements.txt not found
    pause
    exit /b 1
)

echo Uninstalling existing OpenCV and PyTorch packages...
pip uninstall -y torch torchvision torchaudio opencv-python opencv-contrib-python opencv-contrib-python-headless

echo.
echo Installing CUDA-enabled PyTorch (this may take a few minutes)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

if errorlevel 1 (
    echo Error: Failed to install CUDA-enabled PyTorch
    echo.
    echo This tool requires GPU acceleration for practical use.
    echo Please check:
    echo   1. Your internet connection
    echo   2. NVIDIA GPU drivers are installed
    echo   3. CUDA 12.4+ compatible GPU
    echo.
    echo For troubleshooting, see README.md
    pause
    exit /b 1
)

echo.
echo Installing other dependencies...
pip install opencv-contrib-python numpy decord imagededup Pillow tqdm

if errorlevel 1 (
    echo Error: Failed to install other packages
    echo Please check your internet connection and try again
    pause
    exit /b 1
)

echo.
echo Verifying CUDA installation...
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

if errorlevel 1 (
    echo Warning: Failed to verify PyTorch installation
)

echo.
echo ===========================================
echo    Setup Complete! (Integrated Version)
echo ===========================================
echo.
echo IMPORTANT: Please check the CUDA verification above.
echo   - Expected: "CUDA available: True" with GPU name displayed
echo   - If "CUDA available: False" - SETUP FAILED
echo.
echo If CUDA is not available, please follow the manual fix in README.md
echo This tool requires GPU acceleration for practical performance.
echo.
echo Integrated Features:
echo   - SliTraNet: Slide transition detection and frame extraction
echo   - image_dupp: Automatic duplicate image removal
echo.
echo Usage:
echo   1. Prepare video file
echo   2. Run run.bat
echo   3. Drag and drop video file
echo.
echo Processing Flow:
echo   Video -^> Slide Detection -^> Frame Extraction -^> Duplicate Removal
echo.
echo Output:
echo   - extracted_frames/: Final images (after duplicate removal)
echo   - extracted_frames/dupp/: Duplicate images
echo   - similarity_groups.txt: Duplicate detection results
echo.
echo For troubleshooting, see README.md
echo.
pause