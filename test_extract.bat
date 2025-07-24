@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

echo ========================================
echo   Frame Extraction Test Tool
echo   decord Verification Tool
echo ========================================
echo.

:: Check arguments
if "%~1"=="" (
    echo Error: Please drag and drop a video file
    echo.
    pause
    exit /b 1
)

:: Get video file path
set "VIDEO_FILE=%~1"

:: Check file exists
if not exist "%VIDEO_FILE%" (
    echo Error: File not found: %VIDEO_FILE%
    echo.
    pause
    exit /b 1
)

:: Check file extension
set "FILE_EXT=%~x1"
if /i not "%FILE_EXT%"==".mp4" (
    if /i not "%FILE_EXT%"==".avi" (
        if /i not "%FILE_EXT%"==".mov" (
            if /i not "%FILE_EXT%"==".mkv" (
                echo Warning: Unsupported file extension (%FILE_EXT%)
                echo Continue? (Y/N^)
                set /p CONTINUE=
                if /i not "!CONTINUE!"=="Y" exit /b 1
            )
        )
    )
)

echo Video file: %VIDEO_FILE%
echo.

:: Execute Python script
echo Running test_extract.py...
python "%~dp0test_extract.py" "%VIDEO_FILE%"

:: Check results
if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo   Frame Extraction Test Completed
    echo ========================================
    echo Please check the results folder:
    echo %~dp1test_extract
) else (
    echo.
    echo ========================================
    echo   Error Occurred
    echo ========================================
    echo Please check the log file for details
)

echo.
pause