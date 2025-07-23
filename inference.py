# -*- coding: utf-8 -*-
"""
SliTraNet簡易推論スクリプト
動画ファイルを入力として、スライド遷移を検出します

使用方法: python inference.py <video_file>
"""

import os
import sys
import random
import argparse
import numpy as np

import torch
import torch.nn as nn

import decord
from decord import VideoReader

from model import *
from test_slide_detection_2d import detect_initial_slide_transition_candidates_resnet2d
from data.data_utils import *
from data.test_video_clip_dataset import BasicTransform, VideoClipTestDataset


def printLog(*args, **kwargs):
    """ログ出力用関数"""
    print(*args, **kwargs)
    with open('inference.log','a', encoding='utf-8') as file:
        print(*args, **kwargs, file=file)


def detect_slide_transitions(pred_feat):
    """スライド遷移検出"""
    activation = nn.Softmax(dim=1)
    pred_labels = activation(pred_feat)
    scores, pred_classes = torch.max(pred_labels, 1)
    return pred_classes, scores


def get_default_config():
    """デフォルト設定を取得"""
    class DefaultConfig:
        def __init__(self):
            # 基本設定
            self.patch_size = 256
            self.load_checkpoint = False
            
            # 2D CNN設定 (Stage 1)
            self.backbone_2D = 'resnet18'
            self.model_path_2D = 'weights/Frame_similarity_ResNet18_gray.pth'
            self.slide_thresh = 8
            self.video_thresh = 13
            self.input_nc = 2
            self.in_gray = True
            
            # 3D CNN設定 (Stage 2&3)
            self.backbone_3D = 'resnet50'
            self.model_path_1 = 'weights/Slide_video_detection_3DResNet50.pth'
            self.model_path_2 = 'weights/Slide_transition_detection_3DResNet50.pth'
            self.temporal_sampling = 1
            self.clip_length = 8
            self.batch_size = 32
            
    return DefaultConfig()


def create_simple_roi(video_path, patch_size):
    """バウンディングボックス情報なしで全画面を対象とするROI設定"""
    vr = VideoReader(video_path)
    frame = vr[0]
    H, W, _ = frame.shape
    
    # 全画面をROIとして設定
    roi = np.array([0, 0, W, H], dtype=np.int32)
    
    # パッチサイズに合わせてリサイズ比率を計算
    scaling_factor = patch_size / max(H, W)
    new_H = int(H * scaling_factor)
    new_W = int(W * scaling_factor)
    load_size = np.array([new_H, new_W], dtype=np.int32)
    
    # ROIもスケーリング
    scaled_roi = (roi * scaling_factor).astype(np.int32)
    
    return roi, scaled_roi, load_size, load_size


def run_inference(video_path):
    """推論実行メイン関数"""
    # GPU使用可能かチェック
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    printLog(f"使用デバイス: {device}")
    
    if device == 'cpu':
        printLog("警告: GPUが使用できません。CPUで実行するため処理が非常に遅くなる可能性があります。")
    
    # 動画ファイル存在チェック
    if not os.path.exists(video_path):
        printLog(f"エラー: 動画ファイルが見つかりません: {video_path}")
        return False
        
    if not is_video_file(video_path):
        printLog(f"エラー: サポートされていない動画形式です: {video_path}")
        return False
    
    # 設定読み込み
    opt = get_default_config()
    
    # 出力ディレクトリ設定（動画と同じフォルダ）
    video_dir = os.path.dirname(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    pred_dir = os.path.join(video_dir, f"{video_name}_results")
    os.makedirs(pred_dir, exist_ok=True)
    
    printLog(f"動画ファイル: {video_path}")
    printLog(f"結果出力先: {pred_dir}")
    
    # ROI設定（全画面対象）
    try:
        roi, scaled_roi, load_size_roi, load_size_full = create_simple_roi(video_path, opt.patch_size)
        printLog(f"動画解像度: {roi[2]}x{roi[3]}")
        printLog(f"処理解像度: {load_size_roi[1]}x{load_size_roi[0]}")
    except Exception as e:
        printLog(f"エラー: 動画の読み込みに失敗しました: {e}")
        return False
    
    try:
        # モデル読み込み
        printLog("モデルを読み込んでいます...")
        
        # Stage 1: 2D CNN
        opt.n_class = 1
        net2d = define_resnet2d(opt)
        net2d = net2d.to(device)
        net2d = loadNetwork(net2d, opt.model_path_2D, checkpoint=opt.load_checkpoint, prefix='')
        net2d.eval()
        printLog("Stage 1 モデル読み込み完了")
        
        # Stage 2: 3D CNN (スライド-動画検出)
        opt.n_class = 3
        net1 = ResNet3d(opt)
        net1 = net1.to(device)
        net1 = loadNetwork(net1, opt.model_path_1, checkpoint=opt.load_checkpoint, prefix='module.')
        net1.eval()
        printLog("Stage 2 モデル読み込み完了")
        
        # Stage 3: 3D CNN (スライド遷移検出)
        opt.n_class = 4
        net2 = ResNet3d(opt)
        net2 = net2.to(device)
        net2 = loadNetwork(net2, opt.model_path_2, checkpoint=opt.load_checkpoint, prefix='module.')
        net2.eval()
        printLog("Stage 3 モデル読み込み完了")
        
    except Exception as e:
        printLog(f"エラー: モデルの読み込みに失敗しました: {e}")
        return False
    
    # decord設定
    decord.bridge.set_bridge('torch')
    
    printLog("推論を開始します...")
    
    # Stage 1: 初期スライド遷移候補検出
    printLog("Stage 1: 初期スライド遷移候補を検出中...")
    predfile = os.path.join(pred_dir, f'{video_name}_results.txt')
    
    try:
        detect_initial_slide_transition_candidates_resnet2d(
            net2d, video_path, video_name, scaled_roi, load_size_roi, pred_dir, opt
        )
        printLog("Stage 1 完了")
    except Exception as e:
        printLog(f"エラー: Stage 1の処理に失敗しました: {e}")
        return False
    
    # Stage 1の結果読み込み
    try:
        slide_ids, slide_frame_ids_1, slide_frame_ids_2 = read_pred_slide_ids_from_file(predfile)
        slide_transition_pairs, frame_types, slide_transition_types = extract_slide_transitions(
            slide_ids, slide_frame_ids_1, slide_frame_ids_2
        )
        printLog(f"スライド遷移候補: {len(slide_transition_pairs)}件")
    except Exception as e:
        printLog(f"エラー: Stage 1の結果読み込みに失敗しました: {e}")
        return False
    
    if len(slide_transition_pairs) == 0:
        printLog("スライド遷移候補が見つかりませんでした。")
        return True
    
    # Stage 2: スライド-動画判定
    printLog("Stage 2: スライド-動画区間を判定中...")
    try:
        full_clip_dataset = VideoClipTestDataset(
            video_path, load_size_full, slide_transition_pairs, 
            opt.patch_size, opt.clip_length, opt.temporal_sampling,
            n_channels=3, transform=BasicTransform(data_shape="CNHW")
        )
        full_clip_loader = torch.utils.data.DataLoader(
            full_clip_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0
        )
        
        slide_video_prediction = dict()
        
        with torch.no_grad():
            for clips, clip_inds, clip_transition_nums in full_clip_loader:
                clips = clips.to(device)
                pred1 = net1(clips)
                pred_classes, scores = detect_slide_transitions(pred1.squeeze(2).squeeze(2).detach().cpu())
                
                transition_nums = torch.unique(clip_transition_nums)
                for transition_no in transition_nums:
                    key = transition_no.numpy().tolist()
                    if key not in slide_video_prediction:
                        slide_video_prediction[key] = []
                    slide_video_prediction[key].append(
                        pred_classes[torch.where(clip_transition_nums==transition_no)[0]].numpy()
                    )
        
        printLog("Stage 2 完了")
    except Exception as e:
        printLog(f"エラー: Stage 2の処理に失敗しました: {e}")
        return False
    
    # Stage 3: スライド遷移検出
    printLog("Stage 3: スライド遷移を検出中...")
    try:
        clip_dataset = VideoClipTestDataset(
            video_path, load_size_roi, slide_transition_pairs,
            opt.patch_size, opt.clip_length, opt.temporal_sampling,
            n_channels=3, transform=BasicTransform(data_shape="CNHW"),
            roi=scaled_roi
        )
        clip_loader = torch.utils.data.DataLoader(
            clip_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0
        )
        
        slide_transition_prediction = dict()
        
        with torch.no_grad():
            for clips, clip_inds, clip_transition_nums in clip_loader:
                clips = clips.to(device)
                pred2 = net2(clips)
                pred_classes, scores = detect_slide_transitions(pred2.squeeze(2).squeeze(2).detach().cpu())
                
                transition_nums = torch.unique(clip_transition_nums)
                for transition_no in transition_nums:
                    key = transition_no.numpy().tolist()
                    if key not in slide_transition_prediction:
                        slide_transition_prediction[key] = []
                    slide_transition_prediction[key].append(
                        pred_classes[torch.where(clip_transition_nums==transition_no)[0]].numpy()
                    )
        
        printLog("Stage 3 完了")
    except Exception as e:
        printLog(f"エラー: Stage 3の処理に失敗しました: {e}")
        return False
    
    # 結果保存
    try:
        logfile_path = os.path.join(pred_dir, f"{video_name}_transitions.txt")
        with open(logfile_path, "w", encoding='utf-8') as f:
            f.write('Transition No, FrameID0, FrameID1\n')
        
        transition_count = 0
        neg_indices = []
        
        for key in slide_transition_prediction.keys():
            slide_transition_pred = np.hstack(slide_transition_prediction[key])
            slide_video_pred = np.hstack(slide_video_prediction[key])
            
            # 動画区間と判定されたものは除外
            if all(slide_transition_pred==3) and all(slide_video_pred==2):
                neg_indices.append(key)
            else:
                transition_count += 1
                pair = slide_transition_pairs[key]
                
                with open(logfile_path, "a", encoding='utf-8') as f:
                    f.write(f"{transition_count}, {int(pair[0])+1}, {int(pair[1])+1}\n")
                
                printLog(f"遷移 {transition_count}: フレーム {int(pair[0])+1} -> {int(pair[1])+1}")
                printLog(f"  遷移判定: {slide_transition_pred}")
                printLog(f"  動画判定: {slide_video_pred}")
        
        printLog(f"\n==== 結果 ====")
        printLog(f"検出されたスライド遷移: {transition_count}件")
        printLog(f"結果ファイル: {logfile_path}")
        printLog(f"ログファイル: inference.log")
        
        return True
        
    except Exception as e:
        printLog(f"エラー: 結果保存に失敗しました: {e}")
        return False


def main():
    if len(sys.argv) != 2:
        print("使用方法: python inference.py <video_file>")
        print("例: python inference.py sample_video.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    # ログファイル初期化
    if os.path.exists('inference.log'):
        os.remove('inference.log')
    
    printLog("=== SliTraNet 推論開始 ===")
    
    # 推論実行
    success = run_inference(video_path)
    
    if success:
        printLog("=== 推論完了 ===")
        sys.exit(0)
    else:
        printLog("=== 推論失敗 ===")
        sys.exit(1)


if __name__ == '__main__':
    main()