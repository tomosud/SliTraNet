# SliTraNet

## 統合版・ローカル実行対応

動画ファイルから自動でスライド遷移検出とフレーム抽出を一貫して実行します。

## 使用方法

### セットアップ
1. `setup.bat`を実行して環境構築

### 基本使用
1. 動画ファイルを`run.bat`にドラッグ&ドロップ
2. 自動で推論→フレーム抽出が実行される

### コマンドライン使用
```bash
# 基本実行
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
├── inference.log           # 実行ログ
└── video_debug/            # ROI可視化画像（--debugオプション時）
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
