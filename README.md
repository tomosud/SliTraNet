# SliTraNet

## 統合版・ローカル実行対応

このリポジトリは元のSliTraNetをローカル環境で簡単に実行できるように改良した統合版です。
動画ファイルから自動でスライド遷移検出とフレーム抽出を一貫して実行します。

### 主な改良点
- **統合処理**: 推論→フレーム抽出を自動実行
- **簡易セットアップ**: setup.batで一発環境構築
- **ドラッグ&ドロップ実行**: run.batで動画から画像まで自動処理
- **ROI指定機能**: 講演動画の演者部分を除外し、スライド部分のみを検出対象に
- **CUDA自動対応**: GPU/CPU環境を自動判定してモデル読み込み
- **400行制限**: 全Pythonファイルを400行以内でモジュラー設計
- **外部引数対応**: ROI座標の動的設定が可能

## 使用方法

### 基本使用（推奨）
1. `setup.bat`を実行して環境構築
2. 動画ファイルを`run.bat`にドラッグ&ドロップ
3. 自動で推論→フレーム抽出が実行される

### コマンドライン使用
```bash
# 基本実行
run.bat "video.mp4"

# または直接Python実行
python main.py "video.mp4"

# ROI座標を指定
python main.py "video.mp4" --roi-left-top 0.2 0.1 --roi-right-bottom 0.9 0.8

# デバッグモード（ROI可視化）
python main.py "video.mp4" --debug
```

### 出力結果
```
動画フォルダ/
├── video.mp4
├── video_results.txt        # スライド遷移検出結果
├── extracted_frames/        # 抽出フレーム画像
│   ├── slide_001_frame_000175_00h05m51.234s.png
│   └── slide_002_frame_000400_00h13m20.123s.png
├── inference.log           # 実行ログ
└── video_debug/            # ROI可視化画像（--debugオプション時）
```

## コーディングルール

### 全般
- **行数制限**: 全Pythonファイルは400行以内
- **文字エンコーディング**: UTF-8
- **エラーハンドリング**: 例外は適切にキャッチしてログ出力
- **依存関係**: モジュール間の循環参照禁止

### ファイル構成
- **main.py**: 統合エントリポイント（引数解析、フロー制御）
- **inference_core.py**: スライド遷移検出のコア処理
- **frame_extractor.py**: フレーム抽出処理
- **utils.py**: 共通ユーティリティ関数

### 命名規則
- **関数名**: snake_case（例：`run_slide_detection`）
- **変数名**: snake_case（例：`video_path`）
- **定数名**: UPPER_CASE（例：`DEFAULT_ROI`）
- **クラス名**: PascalCase（例：`DefaultConfig`）

### ログ出力
```python
import logging
logger = logging.getLogger(__name__)

# 情報ログ
logger.info("処理開始")

# エラーログ
logger.error("エラー内容", exc_info=True)

# 後方互換性のため printLog() も使用可能
printLog("ログメッセージ")
```

### エラーハンドリング
```python
try:
    # 処理内容
    result = process_video(video_path)
    return True, result
except Exception as e:
    logger.error(f"処理失敗: {e}", exc_info=True)
    return False, None
```

### ファイル操作
```python
# パス操作
import os
from pathlib import Path

# 絶対パス使用
video_path = os.path.abspath(args.video_file)

# ディレクトリ作成
os.makedirs(output_dir, exist_ok=True)

# ファイル存在確認
if not os.path.exists(file_path):
    return False, f"File not found: {file_path}"
```

### 引数処理
```python
def setup_argument_parser():
    parser = argparse.ArgumentParser(
        description='明確な説明',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('video_file', help='動画ファイルのパス')
    return parser
```

### 戻り値の統一
```python
# 成功・失敗の統一形式
def process_function():
    try:
        # 処理
        return True, result_data
    except Exception as e:
        logger.error(f"処理失敗: {e}")
        return False, None

# 使用例
success, data = process_function()
if not success:
    return False, "処理に失敗しました"
```

## 開発・追加実装時の注意

### 機能追加時
1. 400行制限を遵守
2. 既存のログシステムを使用
3. エラーハンドリングの統一形式を維持
4. 引数の妥当性検証を実装

### テスト実行
```bash
# 統合動作テスト
python main.py "test_video.mp4" --debug

# 依存関係チェック
python -c "from utils import check_dependencies; print(check_dependencies())"
```

### トラブルシューティング
- ログファイル: `inference.log`
- GPU使用不可の場合はCPU実行（警告表示）
- ffmpeg未インストールの場合はエラー停止

詳細な改良履歴は[PLAN.md](PLAN.md)を参照してください。

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
