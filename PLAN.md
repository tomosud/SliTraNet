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
- ✅ GPU/CPU自動選択とCUDA対応
- ✅ ROI（Region of Interest）指定機能
- ✅ エラーハンドリングとログ出力
- ✅ 互換性修正（numpy.float廃止対応、OpenCV tensor変換）

## 最新の機能追加・修正

### ROI指定機能
- **目的**: 講演動画で演者の動きを除外し、スライド部分のみを検出対象にする
- **実装**: 正規化座標(0.0-1.0)でAABB指定
- **デフォルト設定**: 左上(0.23, 0.13) 右下(0.97, 0.88)
- **使用方法**: 
  ```bash
  python inference.py video.mp4 --roi-left-top 0.23 0.13 --roi-right-bottom 0.97 0.88
  ```
- **解像度取得**: ffmpegを使用して動画解像度を正確に取得

### CUDA対応強化
- **requirements.txt**: CUDA 12.4版PyTorchに対応（CUDA 12.9環境で動作確認済み）
- **setup.bat**: CPU版PyTorchを強制アンインストールしてGPU版を再インストール
- **model.py**: CPU/GPU環境の自動判定とmap_location設定

### 互換性修正
- **numpy.float廃止**: `data/data_utils.py`で`np.float` → `float`に修正
- **OpenCV tensor変換**: `test_slide_detection_2d.py`でPyTorchテンサーをnumpy配列に変換
- **crop_frame関数**: 幅・高さ形式と座標形式の両方に対応

## トラブルシューティング

### よくある問題と解決方法
1. **"ModuleNotFoundError: No module named 'numpy'"**
   - 解決: `setup.bat`を再実行してvenv環境を再構築
   
2. **"Torch not compiled with CUDA enabled"**
   - 解決: requirements.txtのCUDAバージョンを確認し、setup.batを再実行
   
3. **"OpenCV resize error"**
   - 解決: PyTorchテンサーとnumpy配列の変換処理を確認（修正済み）

4. **検出される遷移が多すぎる（800+件など）**
   - 解決: ROI指定でスライド部分のみを対象にする（デフォルト設定済み）

## システム要件・制約
- **CUDA環境**: GPU使用推奨（CPUでも動作するが非常に遅い）
- **ffmpeg**: 動画解像度取得に使用（システムPATHに設定必要）
- **メモリ**: 大容量動画の場合、GPU/システムメモリ不足の可能性
- **動画形式**: .mp4, .avi, .mov, .m4v（decordライブラリ依存）

## ファイル構成
```
SliTraNet/
├── setup.bat              # 環境構築スクリプト
├── run_inference.bat       # 推論実行バッチ
├── inference.py            # メイン推論スクリプト（ROI対応）
├── model.py               # モデル定義・読み込み（CUDA対応）
├── requirements.txt        # 依存関係（CUDA版PyTorch）
├── data/
│   ├── data_utils.py      # データ処理ユーティリティ（互換性修正済み）
│   └── test_video_clip_dataset.py
├── backbones/             # CNN バックボーン
├── weights/               # 事前訓練済みモデル
└── PLAN.md               # このファイル
```