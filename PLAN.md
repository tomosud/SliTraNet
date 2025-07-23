# SliTraNet セットアップ・推論実行システム 実装計画

## プロジェクト概要
SliTraNetは講義動画のスライド遷移を自動検出するCNNベースのシステムです。
本実装では、簡単に環境構築から推論実行までを行えるツール群を作成します。

## システム構成

### 1. 現在の構成要素
- **モデル**: 3つの事前訓練済みモデル（weights/フォルダ内）
  - `Frame_similarity_ResNet18_gray.pth` (2D CNN - Stage 1)
  - `Slide_video_detection_3DResNet50.pth` (3D CNN - Stage 2)
  - `Slide_transition_detection_3DResNet50.pth` (3D CNN - Stage 3)
- **依存関係**: requirements.txt（torch, torchvision, opencv-contrib-python-headless, numpy, decord）
- **コアモジュール**: model.py, data/data_utils.py, backbones/, test_SliTraNet.py

### 2. 作成予定ファイル

#### A. setup.bat
```batch
@echo off
echo SliTraNet環境セットアップを開始します...
python -m venv venv
call venv\Scripts\activate.bat
pip install -r requirements.txt
echo セットアップ完了！
pause
```

#### B. inference.py
- 動画ファイルのドラッグ&ドロップ対応
- 単一動画ファイルの推論実行
- 元のtest_SliTraNet.pyを簡素化・最適化
- デフォルト設定でバウンディングボックス不要の自動検出モード
- 結果を動画と同じフォルダに出力

#### C. run_inference.bat
```batch
@echo off
call venv\Scripts\activate.bat
python inference.py %1
pause
```

## 実装方針

### 1. 簡素化のポイント
- **バウンディングボックス不要**: 全画面を対象とした推論
- **フォルダ構造簡素化**: 動画ファイルと同じディレクトリに結果出力
- **設定の自動化**: ハードコードされたパラメータを最適化

### 2. 推論プロセス
1. **入力**: 動画ファイル（.mp4, .avi, .mov, .m4v対応）
2. **Stage 1**: 2D CNNで初期スライド遷移候補検出
3. **Stage 2**: 3D CNNでスライド-動画区間判定
4. **Stage 3**: 3D CNNでスライド遷移検出
5. **出力**: 
   - `{動画名}_transitions.txt`: 検出された遷移のフレーム番号
   - `{動画名}_results.txt`: Stage 1の詳細結果
   - `output.out`: 実行ログ

### 3. 技術的考慮事項
- **GPU対応**: CUDA利用可能時は自動でGPU使用
- **メモリ最適化**: バッチサイズの動的調整
- **エラーハンドリング**: 不正な動画形式やCUDA環境のエラー処理
- **依存関係**: 既存のmodel.py, data_utils.pyをそのまま利用

## 使用方法（予定）
1. `setup.bat`を実行して環境構築
2. 動画ファイルを`run_inference.bat`にドラッグ&ドロップ
3. 推論結果が動画と同じフォルダに出力される

## 質問・確認事項
1. 事前学習済みモデルはweights/フォルダに既に存在していますか？
2. 動画のバウンディングボックス情報は不要で、全画面を対象とした推論で良いですか？
3. 成果物の出力形式はテキストファイル（遷移フレーム番号）で良いですか？
4. 特定の動画形式の対応要件はありますか？

## リスク・制約
- CUDA環境が必要（GPUなしでも動作するがCPUのみでは非常に遅い）
- 大容量動画ファイルの場合、メモリ不足の可能性
- decordライブラリの動画コーデック対応範囲に依存