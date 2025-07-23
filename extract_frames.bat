@echo off
setlocal enabledelayedexpansion

echo ===========================================
echo    Middle Frame Extraction Tool
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
if "%~2"=="" (
    echo Error: 2 files required
    echo.
    echo Usage:
    echo   1. Drag and drop 2 files to this bat file:
    echo      - transitions.txt or results.txt
    echo      - video file ^(.mp4^)
    echo   2. Files can be dropped in any order
    echo   3. Extracts middle frame from long slides ^(30+ frame difference^)
    echo   4. Output includes timestamps based on video frame rate
    echo.
    pause
    exit /b 1
)

REM Check file existence
if not exist "%~1" (
    echo Error: File not found
    echo Path: %~1
    echo.
    pause
    exit /b 1
)

if not exist "%~2" (
    echo Error: File not found
    echo Path: %~2
    echo.
    pause
    exit /b 1
)

REM Identify file types
set "TRANSITIONS_FILE="
set "VIDEO_FILE="

REM Check for transitions.txt or results.txt in file 1
echo "%~1" | find /i "transitions.txt" >nul
if !errorlevel! equ 0 (
    set "TRANSITIONS_FILE=%~1"
) else (
    echo "%~1" | find /i "results.txt" >nul
    if !errorlevel! equ 0 (
        set "TRANSITIONS_FILE=%~1"
    ) else (
        echo "%~1" | find /i ".mp4" >nul
        if !errorlevel! equ 0 (
            set "VIDEO_FILE=%~1"
        )
    )
)

REM Check for transitions.txt or results.txt in file 2  
echo "%~2" | find /i "transitions.txt" >nul
if !errorlevel! equ 0 (
    set "TRANSITIONS_FILE=%~2"
) else (
    echo "%~2" | find /i "results.txt" >nul
    if !errorlevel! equ 0 (
        set "TRANSITIONS_FILE=%~2"
    ) else (
        echo "%~2" | find /i ".mp4" >nul
        if !errorlevel! equ 0 (
            set "VIDEO_FILE=%~2"
        )
    )
)

REM Validate required files
if "!TRANSITIONS_FILE!"=="" (
    echo Error: transitions.txt or results.txt file not found
    echo Dropped files:
    echo   1: %~1
    echo   2: %~2
    echo.
    pause
    exit /b 1
)

if "!VIDEO_FILE!"=="" (
    echo Error: .mp4 video file not found
    echo Dropped files:
    echo   1: %~1
    echo   2: %~2
    echo.
    pause
    exit /b 1
)

echo Detected files:
echo   Transitions: !TRANSITIONS_FILE!
echo   Video: !VIDEO_FILE!
echo.

REM Check virtual environment
if not exist "venv\Scripts\activate.bat" (
    echo Warning: Virtual environment not found in !CD!
    echo Using system Python...
    echo.
    goto skip_venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo Error: Failed to activate virtual environment
    pause
    exit /b 1
)
echo Virtual environment activated
echo.

:skip_venv

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

REM Run frame extraction
echo Starting batch middle frame extraction...
echo Note: Extracting middle frames in batches of 10 for optimal performance
echo Processing time depends on video size and number of valid slides
echo.

python extract_frames.py "!TRANSITIONS_FILE!" "!VIDEO_FILE!"
set extraction_result=!errorlevel!

REM Display results
echo.
if !extraction_result! equ 0 (
    echo ===========================================
    echo    Middle Frame Extraction Complete!
    echo ===========================================
    echo.
    for %%i in ("!TRANSITIONS_FILE!") do set "OUTPUT_DIR=%%~dpi"
    echo Results saved to: !OUTPUT_DIR!\extracted_frames
    echo Middle frames extracted in batches for optimal performance
    echo Files include timestamp information based on video FPS
    echo.
) else (
    echo ===========================================
    echo    Extraction Failed
    echo ===========================================
    echo.
    echo Please check the error messages above
    echo Possible causes:
    echo   - No slides with 30+ frame difference found
    echo   - Video file format not supported
    echo   - ffmpeg/ffprobe not available
    echo.
)

pause