@echo off
setlocal enabledelayedexpansion

echo ===========================================
echo    SliTraNet Inference
===========================================
echo.

REM Get the directory where this bat file is located
set "SCRIPT_DIR=%~dp0"
echo Script directory: !SCRIPT_DIR!

REM Change to SliTraNet directory
cd /d "!SCRIPT_DIR!"
echo Changed to: !CD!
echo.

REM Check arguments
if "%~1"=="" (
    echo Error: No video file specified
    echo.
    echo Usage:
    echo   1. Drag and drop video file to this bat file
    echo   2. Or command line: run_inference.bat "video_file.mp4"
    echo.
    echo Supported formats: .mp4, .avi, .mov, .m4v
    echo.
    pause
    exit /b 1
)

set "VIDEO_FILE=%~1"
echo Input video: !VIDEO_FILE!
echo.

REM Check file existence
if not exist "!VIDEO_FILE!" (
    echo Error: Video file not found
    echo Path: !VIDEO_FILE!
    echo.
    pause
    exit /b 1
)

REM Check virtual environment
if not exist "venv\Scripts\activate.bat" (
    echo Error: Virtual environment not found in !CD!
    echo Please run setup.bat first
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo Error: Failed to activate virtual environment
    pause
    exit /b 1
)

REM Run inference
echo.
echo Starting inference...
echo Note: Large video files may take a long time to process
echo.

python inference.py "!VIDEO_FILE!"
set inference_result=!errorlevel!

REM Display results
echo.
if !inference_result! equ 0 (
    echo ===========================================
    echo    Inference Complete!
    echo ===========================================
    echo.
    echo Results saved in the same folder as video:
    echo   - {video_name}_results/{video_name}_transitions.txt : Detected transitions
    echo   - {video_name}_results/{video_name}_results.txt    : Stage1 details
    echo   - inference.log                                    : Execution log
    echo.
) else (
    echo ===========================================
    echo    Inference Failed
    echo ===========================================
    echo.
    echo Check inference.log for error details
    echo.
)

pause