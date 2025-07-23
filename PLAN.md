# SliTraNet ローカル環境セットアップ・推論実行システム

## プロジェクト概要
SliTraNetは講義動画のスライド遷移を自動検出するCNNベースのシステムです。
元のリポジトリをベースに、ローカル環境で簡単に動作するよう改良した推論システムです。

## システム構成

### 1. 現在の構成要素
- **モデル**: 3つの事前訓練済みモデル（weights/フォルダ内）
  - `Frame_similarity_ResNet18_gray.pth` (2D CNN - Stage 1)
  - `Slide_video_detection_3DResNet50.pth` (3D CNN - Stage 2)
  - `Slide_transition_detection_3DResNet50.pth` (3D CNN - Stage 3)
- **依存関係**: requirements.txt（torch, torchvision, opencv-contrib-python-headless, numpy, decord）
- **コアモジュール**: model.py, data/data_utils.py, backbones/, test_SliTraNet.py

### 2. 実装済みファイル

#### A. setup.bat
- Python仮想環境の作成と有効化
- 依存関係の自動インストール（requirements.txt）
- エラーハンドリングと詳細な進行状況表示
- 使用方法の案内表示

#### B. inference.py
- 単一動画ファイルからのスライド遷移検出
- GPU/CPU自動選択（CUDA対応）
- バウンディングボックス不要の全画面対象推論
- 3段階のCNN処理（2D CNN → 3D CNN × 2）
- 詳細なログ出力とエラーハンドリング
- 結果を動画と同じフォルダに出力

#### C. run_inference.bat
- 仮想環境の自動有効化
- 動画ファイルのドラッグ&ドロップ対応
- 推論実行とエラー表示

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

## 使用方法
1. `setup.bat`を実行して環境構築
2. 動画ファイルを`run_inference.bat`にドラッグ&ドロップ
3. 推論結果が動画と同じフォルダの`{動画名}_results`フォルダに出力される

## 実装状況
- ✅ 環境セットアップスクリプト（setup.bat）
- ✅ 推論実行スクリプト（inference.py）
- ✅ 実行用バッチファイル（run_inference.bat）
- ✅ 全画面対象の自動推論
- ✅ GPU/CPU自動選択
- ✅ エラーハンドリングとログ出力

## リスク・制約
- CUDA環境が必要（GPUなしでも動作するがCPUのみでは非常に遅い）
- 大容量動画ファイルの場合、メモリ不足の可能性
- decordライブラリの動画コーデック対応範囲に依存