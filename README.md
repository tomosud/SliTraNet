# SliTraNet - Integrated Video Slide Extraction Tool

## 統合版 (image_dupp機能統合済み)

動画ファイルから自動でスライド遷移検出・フレーム抽出・重複画像除去を一貫して実行します。

## 概要

- **スライド遷移検出**: 深層学習モデルによる高精度なスライド区間検出
- **フレーム抽出**: 各スライドの代表フレームを効率的に抽出  
- **重複画像除去**: dHashアルゴリズムによる類似画像の自動除去

## 処理フロー

```
動画ファイル → スライド遷移検出 → フレーム抽出 → 重複画像除去 → 最終結果
```

## 必要環境

- **Python**: 3.8以上
- **GPU**: NVIDIA GPU (CUDA対応) **必須**
- **CUDA**: 12.4以上 **必須**  
- **メモリ**: 8GB以上推奨
- **ffmpeg**: システムにインストール済み

**⚠️ 重要**: このツールはGPUアクセラレーションが必須です。CPU環境では実用的な処理時間になりません。

## インストール手順

### 1. 初回設定

```cmd
setup.bat
```

**⚠️ 重要: CUDA PyTorchのインストールについて**

setup.batは自動でCUDA対応PyTorchをインストールします。CUDA版のインストールに失敗した場合、セットアップはエラーで停止します。GPU環境が必須のため、CPU版は自動インストールされません。

### 2. CUDA対応の確認

インストール後、以下のコマンドでCUDA対応を確認してください:

```cmd
venv\Scripts\activate
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('PyTorch version:', torch.__version__)"
```

**期待される出力:**
```
CUDA available: True
PyTorch version: 2.6.0+cu124
```

### 3. CUDA問題のトラブルシューティング

もし `CUDA available: False` または `PyTorch version: 2.x.x+cpu` と表示される場合:

#### 手動修正手順:

```cmd
# 仮想環境をアクティベート
venv\Scripts\activate

# 既存のPyTorchをアンインストール
pip uninstall -y torch torchvision torchaudio

# CUDA版PyTorchを手動インストール
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 確認
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

#### 一般的な問題と解決策:

| 問題 | 原因 | 解決策 |
|------|------|--------|
| `CUDA available: False` | CPU版PyTorchがインストールされた | 上記手動修正手順を実行 |
| `Torch not compiled with CUDA enabled` | CPU版PyTorchがインストールされた | 上記手動修正手順を実行 |
| GPUメモリ不足エラー | GPUメモリが不足 | 他のGPUプロセスを終了 |
| CUDAバージョン不適合 | 古いCUDAドライバ | NVIDIAドライバを最新に更新 |

#### CUDA環境確認コマンド:

```cmd
# GPUとCUDAドライバ確認
nvidia-smi

# CUDAツールキット確認 (インストール済みの場合)
nvcc --version
```

## 使用方法

### 基本的な使用方法

1. **動画ファイルを準備**
   - 対応形式: .mp4, .avi, .mov, .m4v, .mkv

2. **処理実行**
   ```cmd
   run.bat
   ```

3. **動画ファイルをドラッグ&ドロップ**
   - run.batに動画ファイルをドラッグ&ドロップ

### コマンドライン実行

```cmd
# 基本実行
python main.py "video.mp4"

# ROI指定実行
python main.py "video.mp4" --roi-left-top 0.2 0.1 --roi-right-bottom 0.9 0.8

# デバッグモード
python main.py "video.mp4" --debug
```

## 出力結果

処理完了後、動画と同じフォルダに以下のファイル・フォルダが生成されます:

```
動画フォルダ/
├── video.mp4                           # 元動画
├── video_results.txt                   # スライド検出結果
├── extracted_frames/                   # 抽出された最終画像
│   ├── slide_001_frame_000123_00h02m15.234s.png
│   ├── slide_002_frame_000456_00h05m42.567s.png
│   └── dupp/                          # 重複除去された画像
│       ├── slide_003_frame_000789.png
│       └── slide_004_frame_001234.png
├── similarity_groups.txt              # 重複検出結果
└── inference.log                      # 処理ログ
```

### ファイル説明

- **extracted_frames/**: 重複除去後の最終スライド画像
- **extracted_frames/dupp/**: 重複と判定され除去された画像
- **similarity_groups.txt**: 重複グループの詳細情報
- **video_results.txt**: スライド遷移検出の詳細結果
- **inference.log**: 処理過程の詳細ログ

## 設定オプション

### ROI設定

デフォルトROI: 左上(0.23, 0.13) 右下(0.97, 0.88)

動画の特性に応じてROIを調整することで検出精度を向上できます:

```cmd
python main.py "video.mp4" --roi-left-top 0.1 0.1 --roi-right-bottom 0.9 0.9
```

### 重複除去閾値

デフォルト類似度閾値: 85%

閾値を変更したい場合は `image_similarity.py` の `THRESHOLD` を編集してください。

## パフォーマンス

### GPU環境での処理時間目安

- **60分講演動画**: 約10-20分
- **GPU VRAM使用量**: 2-4GB
- **システムRAM使用量**: 4-8GB

**注意**: CPU環境での実行は非現実的な処理時間となるため、サポートしていません。

## トラブルシューティング

### よくある問題

1. **「ffmpeg/ffprobe not found」エラー**
   - ffmpegをシステムPATHに追加してください

2. **GPUメモリ不足**
   - 他のGPUプロセスを終了してください
   - タスクマネージャーでGPU使用率を確認

3. **処理が非常に遅い**
   - CUDA対応の確認 (上記手順参照)
   - GPU使用率の確認
   - CPUで動作している場合は即座に停止してCUDA環境を修正

4. **重複除去が機能しない**
   - extracted_framesフォルダに十分な画像があるか確認
   - similarity_groups.txtの内容を確認

### ログ確認

詳細なエラー情報は `inference.log` で確認できます:

```cmd
type inference.log
```

## ファイル構成

- **main.py**: 統合エントリポイント（引数解析、フロー制御）
- **inference_core.py**: スライド遷移検出のコア処理
- **frame_extractor.py**: フレーム抽出処理
- **utils.py**: 共通ユーティリティ関数
- **run.bat**: 統合バッチファイル

## コーディングルール

### 全般
- **行数制限**: 全Pythonファイルは400行以内
- **エラーハンドリング**: 統一形式 `return True, result` / `return False, None`
- **ログ出力**: `logger.info()` / `logger.error()` 使用

### 命名規則
- **関数名**: snake_case
- **変数名**: snake_case  
- **定数名**: UPPER_CASE
- **クラス名**: PascalCase

### 開発時の注意
1. 400行制限を遵守
2. 既存のログシステムを使用
3. エラーハンドリングの統一形式を維持

詳細は[PLAN.md](PLAN.md)を参照してください。

---

## Original SliTraNet
Automatic Detection of Slide Transitions in Lecture Videos using Convolutional Neural Networks

This is the source code to the conference article "SliTraNet: Automatic Detection of Slide Transitions in Lecture Videos using Convolutional Neural Networks" published at OAGM Workshop 2021.

If you use the code, please cite our [paper](https://openlib.tugraz.at/download.php?id=621f329186973&location=browse) ([arxiv](https://arxiv.org/pdf/2202.03540.pdf))

	  @InProceedings{sindel2022slitranet,
		title={SliTraNet: Automatic Detection of Slide Transitions in Lecture Videos using Convolutional Neural Networks},
		author={Aline Sindel and Abner Hernandez and Seung Hee Yang and Vincent Christlein and Andreas Maier},
		year={2022},
		booktitle={Proceedings of the OAGM Workshop 2021},
		doi={10.3217/978-3-85125-869-1-10},
		pages={59-64}		
	  }

## Requirements

Install the requirements using pip or conda (python 3):
- torch >= 1.7
- torchvision
- opencv-contrib-python-headless
- numpy
- decord

## Usage

### Data

The dataset needs to be in the following folder structure:
- Video files in: "/videos/PHASE/", where PHASE is "train", "val" or "test".
- Bounding box labels in: "/videos/PHASE_bounding_box_list.txt"

Bounding box labels define the rectangle of the slide area in the format: Videoname,x0,y0,x1,y1

Here one example test_bounding_box_list.txt file (the header needs to be included):  
Video,x0,y0,x1,y1  
Architectures_1,38,57,1306,1008  
Architectures_2,38,57,1306,1008  


### Pretrained weights

The pretrained weights of SliTraNet from the paper can be downloaded [here](https://drive.google.com/drive/folders/1aQDVplbbpt-zgH2O1q7685AZ1hl0BsVV?usp=sharing).
Move them into the folder: "/weights"

### SliTraNet Inference: 

Run test_SliTraNet.py 

Some settings have to be specified, as described in the python file, such as the dataset and output folders and model paths.

Stage 1 of SliTraNet can also be applied separately (see test_slide_detection_2d.py) and afterwards the results can be loaded in test_SliTraNet.py.


@author Aline Sindel
