# SliTraNet

## 日本語版改良・ローカル実行対応

このリポジトリは元のSliTraNetをローカル環境で簡単に実行できるように改良したフォーク版です。

### 主な改良点
- **簡易セットアップ**: setup.batで一発環境構築
- **ドラッグ&ドロップ実行**: run_inference.batで動画ファイルを簡単実行
- **ROI指定機能**: 講演動画の演者部分を除外し、スライド部分のみを検出対象に
- **CUDA自動対応**: GPU/CPU環境を自動判定してモデル読み込み
- **互換性修正**: 最新PyTorch/numpy環境での動作保証

### 使用方法
1. `setup.bat`を実行して環境構築
2. 動画ファイルを`run_inference.bat`にドラッグ&ドロップ
3. 結果が動画と同じフォルダの`{動画名}_results`に出力

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
