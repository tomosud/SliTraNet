@echo off
setlocal enabledelayedexpansion

echo ===========================================
echo    SliTraNet Integrated Processing
echo    (Video -^> Slide Detection -^> Frame Extraction)
echo ===========================================
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
    echo   2. Or command line: run.bat "video_file.mp4"
    echo.
    echo Supported formats: .mp4, .avi, .mov, .m4v, .mkv
    echo.
    echo Processing steps:
    echo   1. Slide transition detection ^(Stage1^)
    echo   2. Middle frame extraction ^(slides with 30+ frames only^)
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

REM Check file extension
set "FILE_EXT="
for %%i in ("!VIDEO_FILE!") do set "FILE_EXT=%%~xi"
echo !FILE_EXT! | findstr /i "\.mp4 \.avi \.mov \.m4v \.mkv" >nul
if !errorlevel! neq 0 (
    echo Error: Unsupported video format
    echo Format: !FILE_EXT!
    echo Supported formats: .mp4, .avi, .mov, .m4v, .mkv
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

REM Check dependencies
echo.
echo Checking dependencies...

REM Check ffmpeg
ffmpeg -version >nul 2>&1
if !errorlevel! neq 0 (
    echo Error: ffmpeg not found
    echo Please make sure ffmpeg is in system PATH
    echo.
    pause
    exit /b 1
)

echo ffmpeg check OK
echo.

REM Set environment for character encoding
set PYTHONIOENCODING=utf-8

REM Run integrated processing
echo ===========================================
echo Starting integrated processing...
echo ===========================================
echo.
echo Step 1: Slide transition detection
echo Step 2: Frame extraction
echo.
echo Note: Large video files may take a long time to process
echo.

python main.py "!VIDEO_FILE!"
set processing_result=!errorlevel!

REM Display results
echo.
if !processing_result! equ 0 (
    echo ===========================================
    echo    Integrated processing completed successfully!
    echo ===========================================
    echo.
    for %%i in ("!VIDEO_FILE!") do (
        set "VIDEO_DIR=%%~dpi"
        set "VIDEO_NAME=%%~ni"
    )
    echo Result files:
    echo   - !VIDEO_NAME!_results.txt     : Slide transition detection results
    echo   - extracted_frames\            : Extracted frame images
    echo   - inference.log                : Execution log
    echo.
    echo Saved in the same folder as video:
    echo   !VIDEO_DIR!
    echo.
) else (
    echo ===========================================
    echo    Error occurred during integrated processing
    echo ===========================================
    echo.
    echo Check inference.log for error details
    echo.
    if !processing_result! equ 1 (
        echo Common error causes:
        echo   - Video file format or corruption
        echo   - GPU/CPU memory shortage
        echo   - Missing or corrupted model files
    ) else if !processing_result! equ 130 (
        echo User interrupted processing ^(Ctrl+C^)
    ) else (
        echo Unexpected error ^(exit code: !processing_result!^)
    )
    echo.
)

pause