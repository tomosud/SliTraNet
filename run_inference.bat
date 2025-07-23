@echo off
setlocal enabledelayedexpansion

echo ===========================================
echo    SliTraNet 推論実行
echo ===========================================
echo.

REM 引数チェック
if "%~1"=="" (
    echo エラー: 動画ファイルが指定されていません
    echo.
    echo 使用方法:
    echo   1. 動画ファイルをこのbatファイルにドラッグ&ドロップ
    echo   2. またはコマンドラインで: run_inference.bat "動画ファイル.mp4"
    echo.
    echo サポートされる形式: .mp4, .avi, .mov, .m4v
    echo.
    pause
    exit /b 1
)

set "VIDEO_FILE=%~1"
echo 入力動画: !VIDEO_FILE!
echo.

REM ファイル存在チェック
if not exist "!VIDEO_FILE!" (
    echo エラー: 指定された動画ファイルが見つかりません
    echo パス: !VIDEO_FILE!
    echo.
    pause
    exit /b 1
)

REM 仮想環境チェック
if not exist "venv\Scripts\activate.bat" (
    echo エラー: 仮想環境が見つかりません
    echo 先にsetup.batを実行してください
    echo.
    pause
    exit /b 1
)

REM 仮想環境アクティベート
echo 仮想環境をアクティベートしています...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo エラー: 仮想環境のアクティベートに失敗しました
    pause
    exit /b 1
)

REM 推論実行
echo.
echo 推論を開始します...
echo ※大きな動画ファイルの場合、処理に時間がかかる場合があります
echo.

python inference.py "!VIDEO_FILE!"
set inference_result=!errorlevel!

REM 結果表示
echo.
if !inference_result! equ 0 (
    echo ===========================================
    echo    推論完了！
    echo ===========================================
    echo.
    echo 結果は動画ファイルと同じフォルダに出力されました:
    echo   - {動画名}_results/{動画名}_transitions.txt : 検出された遷移
    echo   - {動画名}_results/{動画名}_results.txt    : Stage1の詳細結果
    echo   - inference.log                           : 実行ログ
    echo.
) else (
    echo ===========================================
    echo    推論失敗
    echo ===========================================
    echo.
    echo inference.logファイルでエラー内容を確認してください
    echo.
)

pause