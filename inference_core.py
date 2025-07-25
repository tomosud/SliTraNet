# -*- coding: utf-8 -*-
"""
SliTraNet推論コア処理
スライド遷移検出のメイン処理を実行
"""

import os
import sys
import logging
import numpy as np
import cv2
import torch
import torch.nn as nn
import decord
from decord import VideoReader

from model import define_resnet2d, ResNet3d, loadNetwork
from slide_detection_2d import detect_initial_slide_transition_candidates_resnet2d
from data.data_utils import read_pred_slide_ids_from_file, extract_slide_transitions
from utils import printLog, get_video_info_ffmpeg


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
            self.slide_thresh = 5  # より多くの候補を検出
            self.video_thresh = 10  # より多くの候補を検出
            self.input_nc = 2
            self.in_gray = True
            
            # 3D CNN設定 (Stage 2&3) - 現在は使用しない
            self.backbone_3D = 'resnet50'
            self.model_path_1 = 'weights/Slide_video_detection_3DResNet50.pth'
            self.model_path_2 = 'weights/Slide_transition_detection_3DResNet50.pth'
            self.temporal_sampling = 1
            self.clip_length = 8
            self.batch_size = 32
            
    return DefaultConfig()


def create_roi_from_normalized(video_path, patch_size, roi_left_top=(0.0, 0.0), roi_right_bottom=(1.0, 1.0)):
    """正規化座標からROIを作成"""
    logger = logging.getLogger(__name__)
    
    # 動画解像度取得
    video_info = get_video_info_ffmpeg(video_path)
    W, H = video_info['width'], video_info['height']
    
    printLog(f"動画解像度: {W}x{H}")
    logger.info(f"動画解像度: {W}x{H}")
    
    # 正規化座標をピクセル座標に変換
    x1 = int(roi_left_top[0] * W)
    y1 = int(roi_left_top[1] * H)
    x2 = int(roi_right_bottom[0] * W)
    y2 = int(roi_right_bottom[1] * H)
    
    # AABB形式 [x, y, width, height]
    roi_width = x2 - x1
    roi_height = y2 - y1
    roi = np.array([x1, y1, roi_width, roi_height], dtype=np.int32)
    
    printLog(f"ROI設定: [{x1}, {y1}, {roi_width}, {roi_height}] (x={x1}-{x2}, y={y1}-{y2})")
    logger.info(f"ROI設定: [{x1}, {y1}, {roi_width}, {roi_height}]")
    
    # パッチサイズに合わせてリサイズ比率を計算
    scaling_factor = patch_size / max(H, W)
    new_H = int(H * scaling_factor)
    new_W = int(W * scaling_factor)
    load_size = np.array([new_H, new_W], dtype=np.int32)
    
    # ROIもスケーリング
    scaled_roi = (roi * scaling_factor).astype(np.int32)
    
    printLog(f"処理解像度: {new_W}x{new_H}")
    logger.info(f"処理解像度: {new_W}x{new_H}")
    
    return roi, scaled_roi, load_size, load_size


def debug_visualize_roi(video_path, roi, scaled_roi, load_size_roi, debug_dir):
    """ROI処理結果を可視化するデバッグ機能"""
    logger = logging.getLogger(__name__)
    
    try:
        printLog("=== ROI可視化デバッグ開始 ===")
        logger.info("ROI可視化デバッグ開始")
        
        # 動画読み込み（元解像度）
        vr_original = VideoReader(video_path)
        total_frames = len(vr_original)
        
        # 動画読み込み（処理解像度）
        vr_resized = VideoReader(video_path, width=load_size_roi[1], height=load_size_roi[0])
        
        # 5箇所のフレームを選択
        sample_indices = [
            0,
            total_frames // 4,
            total_frames // 2,
            total_frames * 3 // 4,
            total_frames - 1
        ]
        
        # デバッグディレクトリ作成
        os.makedirs(debug_dir, exist_ok=True)
        printLog(f"デバッグ出力先: {debug_dir}")
        logger.info(f"デバッグ出力先: {debug_dir}")
        
        for i, frame_idx in enumerate(sample_indices):
            try:
                # 元解像度フレーム取得
                original_frame = vr_original[frame_idx].numpy()
                resized_frame = vr_resized[frame_idx].numpy()
                
                # ROI境界を描画
                roi_boundary_img = original_frame.copy()
                cv2.rectangle(roi_boundary_img, 
                             (roi[0], roi[1]), 
                             (roi[0] + roi[2], roi[1] + roi[3]), 
                             (0, 255, 0), 3)
                
                # ROI部分を切り出し
                roi_cropped = original_frame[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
                resized_roi_cropped = resized_frame[scaled_roi[1]:scaled_roi[1]+scaled_roi[3], 
                                                 scaled_roi[0]:scaled_roi[0]+scaled_roi[2]]
                
                # 画像保存（日本語パス対応）
                def save_image_jp(filepath, img):
                    try:
                        ext = os.path.splitext(filepath)[1]
                        result, encoded_img = cv2.imencode(ext, img)
                        if result:
                            encoded_img.tofile(filepath)
                            return True
                        return False
                    except:
                        return False
                
                file1 = os.path.join(debug_dir, f"frame_{i+1:02d}_original_with_roi.png")
                file2 = os.path.join(debug_dir, f"frame_{i+1:02d}_roi_cropped_original.png")
                file3 = os.path.join(debug_dir, f"frame_{i+1:02d}_roi_cropped_resized.png")
                
                success1 = save_image_jp(file1, cv2.cvtColor(roi_boundary_img, cv2.COLOR_RGB2BGR))
                success2 = save_image_jp(file2, cv2.cvtColor(roi_cropped, cv2.COLOR_RGB2BGR))
                success3 = save_image_jp(file3, cv2.cvtColor(resized_roi_cropped, cv2.COLOR_RGB2BGR))
                
                printLog(f"保存結果 フレーム{frame_idx} ({i+1}/5): {success1}, {success2}, {success3}")
                
            except Exception as frame_error:
                printLog(f"フレーム{frame_idx}処理エラー: {frame_error}")
                logger.error(f"フレーム{frame_idx}処理エラー: {frame_error}")
        
        printLog(f"ROI可視化画像を保存しました: {debug_dir}")
        printLog("=== ROI可視化デバッグ完了 ===")
        logger.info("ROI可視化デバッグ完了")
        
    except Exception as e:
        printLog(f"ROI可視化エラー: {e}")
        logger.error(f"ROI可視化エラー: {e}")


def load_inference_model(device, opt):
    """推論モデルの読み込み"""
    logger = logging.getLogger(__name__)
    
    try:
        printLog("モデルを読み込んでいます...")
        logger.info("モデル読み込み開始")
        
        # Stage 1: 2D CNN
        opt.n_class = 1
        net2d = define_resnet2d(opt)
        net2d = net2d.to(device)
        net2d = loadNetwork(net2d, opt.model_path_2D, checkpoint=opt.load_checkpoint, prefix='')
        net2d.eval()
        
        printLog("Stage 1 モデル読み込み完了")
        logger.info("Stage 1 モデル読み込み完了")
        
        return net2d, True
        
    except Exception as e:
        printLog(f"エラー: モデルの読み込みに失敗しました: {e}")
        logger.error(f"モデル読み込み失敗: {e}")
        return None, False


def run_slide_detection(video_path, roi_left_top=(0.0, 0.0), roi_right_bottom=(1.0, 1.0), debug_mode=False):
    """スライド遷移検出のメイン処理"""
    logger = logging.getLogger(__name__)
    
    try:
        # GPU使用可能かチェック
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        printLog(f"使用デバイス: {device}")
        logger.info(f"使用デバイス: {device}")
        
        if device == 'cpu':
            printLog("警告: GPUが使用できません。CPUで実行するため処理が非常に遅くなる可能性があります。")
            logger.warning("GPU使用不可、CPU実行")
        
        # 設定読み込み
        opt = get_default_config()
        
        # 出力設定
        video_dir = os.path.dirname(video_path)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        pred_dir = video_dir
        
        printLog(f"動画ファイル: {video_path}")
        printLog(f"結果出力先: {pred_dir}")
        logger.info(f"動画ファイル: {video_path}")
        logger.info(f"結果出力先: {pred_dir}")
        
        # ROI設定
        roi, scaled_roi, load_size_roi, load_size_full = create_roi_from_normalized(
            video_path, opt.patch_size, roi_left_top, roi_right_bottom
        )
        
        # decord設定
        decord.bridge.set_bridge('torch')
        
        # デバッグモード時のROI可視化
        if debug_mode:
            debug_dir = os.path.join(pred_dir, f"{video_name}_debug")
            debug_visualize_roi(video_path, roi, scaled_roi, load_size_roi, debug_dir)
        
        # モデル読み込み
        net2d, model_success = load_inference_model(device, opt)
        if not model_success:
            return False, None
        
        # Stage 1: 初期スライド遷移候補検出
        printLog("Stage 1: 初期スライド遷移候補を検出中...")
        logger.info("Stage 1開始")
        
        predfile = os.path.join(pred_dir, f'{video_name}_results.txt')
        
        detect_initial_slide_transition_candidates_resnet2d(
            net2d, video_path, video_name, scaled_roi, load_size_roi, pred_dir, opt
        )
        
        printLog("Stage 1 完了")
        logger.info("Stage 1完了")
        
        # Stage 1の結果読み込みと検証
        slide_ids, slide_frame_ids_1, slide_frame_ids_2 = read_pred_slide_ids_from_file(predfile)
        slide_transition_pairs, frame_types, slide_transition_types = extract_slide_transitions(
            slide_ids, slide_frame_ids_1, slide_frame_ids_2
        )
        
        printLog(f"スライド遷移候補: {len(slide_transition_pairs)}件")
        logger.info(f"スライド遷移候補: {len(slide_transition_pairs)}件")
        
        if len(slide_transition_pairs) == 0:
            printLog("スライド遷移候補が見つかりませんでした。")
            logger.warning("スライド遷移候補なし")
        
        # _results.txt出力完了
        printLog(f"Stage 1完了: {predfile} を出力しました")
        logger.info(f"結果ファイル出力: {predfile}")
        
        return True, predfile
        
    except Exception as e:
        printLog(f"エラー: スライド遷移検出に失敗しました: {e}")
        logger.error(f"スライド遷移検出失敗: {e}", exc_info=True)
        return False, None