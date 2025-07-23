@echo off
echo ===========================================
echo    SliTraNet 環境セットアップ
echo ===========================================
echo.

echo 仮想環境を作成しています...
python -m venv venv
if errorlevel 1 (
    echo エラー: 仮想環境の作成に失敗しました
    echo Python 3.x がインストールされていることを確認してください
    pause
    exit /b 1
)

echo.
echo 仮想環境をアクティベートしています...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo エラー: 仮想環境のアクティベートに失敗しました
    pause
    exit /b 1
)

echo.
echo 依存パッケージをインストールしています...
pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo エラー: パッケージのインストールに失敗しました
    pause
    exit /b 1
)

echo.
echo ===========================================
echo    セットアップ完了！
echo ===========================================
echo.
echo 使用方法:
echo   1. 動画ファイルを用意します
echo   2. run_inference.bat を実行します
echo   3. 動画ファイルのパスを引数として渡します
echo.
echo 例: run_inference.bat "C:\path\to\video.mp4"
echo.
pause