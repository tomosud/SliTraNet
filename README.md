# SliTraNet - 動画スライド自動抽出ツール

動画ファイルから自動でスライド遷移を検出し、フレームを抽出、重複画像を除去するツールです。

## 概要

- **スライド遷移検出**: 深層学習による高精度なスライド区間検出
- **フレーム抽出**: 各スライドの代表フレームを自動抽出  
- **重複画像除去**: 類似画像を自動で除去

## クイックスタート

### 1. 初回セットアップ
```cmd
setup.bat
```

### 2. 動画処理
1. `run.bat` を実行
2. 動画ファイル（.mp4, .avi, .mov, .m4v, .mkv）をドラッグ&ドロップ
3. `extracted_frames/` フォルダで結果を確認

### 3. 出力結果
```
動画フォルダ/
├── video_results.txt      # スライド検出結果
├── extracted_frames/      # 最終画像（重複除去後）
│   └── dupp/             # 重複と判定された画像
├── similarity_groups.txt  # 重複検出詳細
└── inference.log          # 処理ログ
```

## 必要環境

- **Python 3.8+**
- **NVIDIA GPU (CUDA必須)** - CPU環境は非対応
- **ffmpeg** - システムPATHに追加
- **8GB以上のRAM推奨**

## トラブルシューティング

### CUDA問題
セットアップ後にCUDA確認:
```cmd
venv\Scripts\activate
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

`False`の場合の修正:
```cmd
venv\Scripts\activate
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### よくある問題
- **ffmpeg not found**: ffmpegをシステムPATHに追加
- **GPUメモリ不足**: 他のGPUプロセスを終了
- **処理が遅い**: CUDA環境を確認（CPU実行は非対応）

### ログ確認
```cmd
type inference.log
```

## 高度な使用法

### コマンドライン実行
```cmd
# 基本実行
python main.py "video.mp4"

# ROI指定実行
python main.py "video.mp4" --roi-left-top 0.2 0.1 --roi-right-bottom 0.9 0.8
```

### 設定調整
- **ROI設定**: デフォルト左上(0.23, 0.13) 右下(0.97, 0.88)
- **重複除去閾値**: `image_similarity.py`の`THRESHOLD`（デフォルト85%）

## パフォーマンス

- **60分動画**: 約10-20分で処理
- **GPU VRAM**: 2-4GB使用
- **RAM**: 4-8GB使用

## ファイル構成

```
SliTraNet/
├── main.py               # 統合エントリポイント
├── run.bat              # バッチ実行ファイル
├── inference_core.py    # スライド検出コア処理
├── slide_detection_2d.py # Stage 1 スライド検出
├── frame_extractor.py   # フレーム抽出処理
├── image_similarity.py  # 重複画像除去
├── model.py             # モデル定義
├── utils.py             # 共通ユーティリティ
├── backbones/           # モデルバックボーン
├── data/                # データ処理モジュール
└── weights/             # 学習済みモデル
```

---

## 開発者向け情報

詳細な実装記録は [PLAN.md](PLAN.md) を参照してください。

## 元論文

This is based on "SliTraNet: Automatic Detection of Slide Transitions in Lecture Videos using Convolutional Neural Networks" (OAGM Workshop 2021).

[Paper](https://openlib.tugraz.at/download.php?id=621f329186973&location=browse) | [arXiv](https://arxiv.org/pdf/2202.03540.pdf)

```bibtex
@InProceedings{sindel2022slitranet,
  title={SliTraNet: Automatic Detection of Slide Transitions in Lecture Videos using Convolutional Neural Networks},
  author={Aline Sindel and Abner Hernandez and Seung Hee Yang and Vincent Christlein and Andreas Maier},
  year={2022},
  booktitle={Proceedings of the OAGM Workshop 2021},
  doi={10.3217/978-3-85125-869-1-10},
  pages={59-64}
}
```
