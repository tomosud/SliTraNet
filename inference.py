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
import subprocess
import json
import cv2

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
            self.slide_thresh = 5  # 8 → 5 (より多くの候補を検出)
            self.video_thresh = 10  # 13 → 10 (より多くの候補を検出)
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


def get_video_resolution_ffmpeg(video_path):
    """ffmpegを使用して動画解像度を取得"""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        
        for stream in data['streams']:
            if stream['codec_type'] == 'video':
                return stream['width'], stream['height']
        
        raise ValueError("Video stream not found")
    except Exception as e:
        printLog(f"ffmpegでの解像度取得に失敗: {e}")
        # フォールバック: decordを使用
        vr = VideoReader(video_path)
        frame = vr[0]
        H, W, _ = frame.shape
        return W, H


def debug_visualize_roi(video_path, roi, scaled_roi, load_size_roi, debug_dir):
    """ROI処理結果を可視化するデバッグ機能"""
    try:
        printLog("=== ROI可視化デバッグ開始 ===")
        
        # 動画読み込み（元解像度）
        vr_original = VideoReader(video_path)
        total_frames = len(vr_original)
        
        # 動画読み込み（処理解像度）
        vr_resized = VideoReader(video_path, width=load_size_roi[1], height=load_size_roi[0])
        
        # 5箇所のフレームを選択（開始、1/4、中央、3/4、終了付近）
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
        
        for i, frame_idx in enumerate(sample_indices):
            try:
                # 元解像度フレーム取得
                original_frame = vr_original[frame_idx].numpy()
                printLog(f"元フレーム形状: {original_frame.shape}")
                
                # 処理解像度フレーム取得
                resized_frame = vr_resized[frame_idx].numpy()
                printLog(f"リサイズフレーム形状: {resized_frame.shape}")
                
                # 1. 元画像にROI境界を描画
                roi_boundary_img = original_frame.copy()
                cv2.rectangle(roi_boundary_img, 
                             (roi[0], roi[1]), 
                             (roi[0] + roi[2], roi[1] + roi[3]), 
                             (0, 255, 0), 3)  # 緑色の枠線
                
                # 2. 元画像からROI部分を切り出し
                roi_cropped = original_frame[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
                printLog(f"ROI切り出し形状: {roi_cropped.shape}")
                
                # 3. 処理解像度からROI部分を切り出し
                resized_roi_cropped = resized_frame[scaled_roi[1]:scaled_roi[1]+scaled_roi[3], 
                                                 scaled_roi[0]:scaled_roi[0]+scaled_roi[2]]
                printLog(f"リサイズROI切り出し形状: {resized_roi_cropped.shape}")
                
                # 画像保存（日本語パス対応）
                file1 = os.path.join(debug_dir, f"frame_{i+1:02d}_original_with_roi.png")
                file2 = os.path.join(debug_dir, f"frame_{i+1:02d}_roi_cropped_original.png")
                file3 = os.path.join(debug_dir, f"frame_{i+1:02d}_roi_cropped_resized.png")
                
                # OpenCVの日本語パス問題を回避
                def save_image_jp(filepath, img):
                    try:
                        # cv2.imencode + np.tofile を使用
                        ext = os.path.splitext(filepath)[1]
                        result, encoded_img = cv2.imencode(ext, img)
                        if result:
                            encoded_img.tofile(filepath)
                            return True
                        return False
                    except:
                        return False
                
                success1 = save_image_jp(file1, cv2.cvtColor(roi_boundary_img, cv2.COLOR_RGB2BGR))
                success2 = save_image_jp(file2, cv2.cvtColor(roi_cropped, cv2.COLOR_RGB2BGR))
                success3 = save_image_jp(file3, cv2.cvtColor(resized_roi_cropped, cv2.COLOR_RGB2BGR))
                
                printLog(f"保存結果 フレーム{frame_idx} ({i+1}/5): {success1}, {success2}, {success3}")
                
            except Exception as frame_error:
                printLog(f"フレーム{frame_idx}処理エラー: {frame_error}")
        
        printLog(f"ROI可視化画像を保存しました: {debug_dir}")
        printLog("=== ROI可視化デバッグ完了 ===")
        
    except Exception as e:
        printLog(f"ROI可視化エラー: {e}")
        import traceback
        printLog(f"トレースバック: {traceback.format_exc()}")


def create_roi_from_normalized(video_path, patch_size, roi_left_top=(0.0, 0.0), roi_right_bottom=(1.0, 1.0)):
    """正規化座標からROIを作成"""
    # 動画解像度取得
    W, H = get_video_resolution_ffmpeg(video_path)
    printLog(f"動画解像度: {W}x{H}")
    
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
    
    # パッチサイズに合わせてリサイズ比率を計算
    scaling_factor = patch_size / max(H, W)
    new_H = int(H * scaling_factor)
    new_W = int(W * scaling_factor)
    load_size = np.array([new_H, new_W], dtype=np.int32)
    
    # ROIもスケーリング
    scaled_roi = (roi * scaling_factor).astype(np.int32)
    
    return roi, scaled_roi, load_size, load_size


def run_inference(video_path, roi_left_top=(0.0, 0.0), roi_right_bottom=(1.0, 1.0)):
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
    
    # 出力ディレクトリ設定（動画と同じフォルダに直接出力）
    video_dir = os.path.dirname(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    pred_dir = video_dir  # 直接動画と同じフォルダに出力
    
    printLog(f"動画ファイル: {video_path}")
    printLog(f"結果出力先: {pred_dir}")
    
    # ROI設定
    try:
        roi, scaled_roi, load_size_roi, load_size_full = create_roi_from_normalized(
            video_path, opt.patch_size, roi_left_top, roi_right_bottom
        )
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
    
    # ===== ROI可視化用デバッグ機能 (一時的) =====
    # 以下のコメントアウトを外すとROI処理結果を可視化できます
    # デバッグディレクトリは一時的に作成
    debug_dir = os.path.join(pred_dir, f"{video_name}_debug")
    debug_visualize_roi(video_path, roi, scaled_roi, load_size_roi, debug_dir)
    
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
    
    # _results.txt出力完了のため、Stage 1で処理終了
    printLog(f"Stage 1完了: {predfile} を出力しました")
    printLog("=== 推論完了 (_results.txtまで) ===")
    return True
    
    # ===== 以下はStage 2・3の処理（_results.txt出力後の追加処理） =====
    # 必要に応じてコメントアウトを外してください
    
    # # Stage 2: スライド-動画判定
    # printLog("Stage 2: スライド-動画区間を判定中...")
    # try:
    #     full_clip_dataset = VideoClipTestDataset(
    #         video_path, load_size_full, slide_transition_pairs, 
    #         opt.patch_size, opt.clip_length, opt.temporal_sampling,
    #         n_channels=3, transform=BasicTransform(data_shape="CNHW")
    #     )
    #     full_clip_loader = torch.utils.data.DataLoader(
    #         full_clip_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0
    #     )
    #     
    #     slide_video_prediction = dict()
    #     
    #     with torch.no_grad():
    #         for clips, clip_inds, clip_transition_nums in full_clip_loader:
    #             clips = clips.to(device)
    #             pred1 = net1(clips)
    #             pred_classes, scores = detect_slide_transitions(pred1.squeeze(2).squeeze(2).detach().cpu())
    #             
    #             transition_nums = torch.unique(clip_transition_nums)
    #             for transition_no in transition_nums:
    #                 key = transition_no.numpy().tolist()
    #                 if key not in slide_video_prediction:
    #                     slide_video_prediction[key] = []
    #                 slide_video_prediction[key].append(
    #                     pred_classes[torch.where(clip_transition_nums==transition_no)[0]].numpy()
    #                 )
    #     
    #     printLog("Stage 2 完了")
    # except Exception as e:
    #     printLog(f"エラー: Stage 2の処理に失敗しました: {e}")
    #     return False
    
    # # Stage 3: スライド遷移検出
    # printLog("Stage 3: スライド遷移を検出中...")
    # try:
    #     clip_dataset = VideoClipTestDataset(
    #         video_path, load_size_roi, slide_transition_pairs,
    #         opt.patch_size, opt.clip_length, opt.temporal_sampling,
    #         n_channels=3, transform=BasicTransform(data_shape="CNHW"),
    #         roi=scaled_roi
    #     )
    #     clip_loader = torch.utils.data.DataLoader(
    #         clip_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0
    #     )
    #     
    #     slide_transition_prediction = dict()
    #     
    #     with torch.no_grad():
    #         for clips, clip_inds, clip_transition_nums in clip_loader:
    #             clips = clips.to(device)
    #             pred2 = net2(clips)
    #             pred_classes, scores = detect_slide_transitions(pred2.squeeze(2).squeeze(2).detach().cpu())
    #             
    #             transition_nums = torch.unique(clip_transition_nums)
    #             for transition_no in transition_nums:
    #                 key = transition_no.numpy().tolist()
    #                 if key not in slide_transition_prediction:
    #                     slide_transition_prediction[key] = []
    #                 slide_transition_prediction[key].append(
    #                     pred_classes[torch.where(clip_transition_nums==transition_no)[0]].numpy()
    #                 )
    #     
    #     printLog("Stage 3 完了")
    # except Exception as e:
    #     printLog(f"エラー: Stage 3の処理に失敗しました: {e}")
    #     return False
    # 
    # # 結果保存（transitions.txtは作成しない）
    # try:
    #     transition_count = 0
    #     neg_indices = []
    #     
    #     for key in slide_transition_prediction.keys():
    #         slide_transition_pred = np.hstack(slide_transition_prediction[key])
    #         slide_video_pred = np.hstack(slide_video_prediction[key])
    #         
    #         # 改善されたフィルタリング条件: 多数決ベース
    #         video_confidence = np.mean(slide_video_pred == 2)  # 動画判定の割合
    #         transition_confidence = np.mean(slide_transition_pred == 3)  # 動画遷移判定の割合
    #         
    #         # より緩い条件で動画区間を除外
    #         if video_confidence > 0.6 and transition_confidence > 0.6:
    #             neg_indices.append(key)
    #             printLog(f"除外: フレーム {int(slide_transition_pairs[key][0])+1} -> {int(slide_transition_pairs[key][1])+1}")
    #             printLog(f"  動画信頼度: {video_confidence:.2f}, 遷移信頼度: {transition_confidence:.2f}")
    #         else:
    #             transition_count += 1
    #             pair = slide_transition_pairs[key]
    #             
    #             printLog(f"遷移 {transition_count}: フレーム {int(pair[0])+1} -> {int(pair[1])+1}")
    #             printLog(f"  遷移判定: {slide_transition_pred}")
    #             printLog(f"  動画判定: {slide_video_pred}")
    #             printLog(f"  動画信頼度: {video_confidence:.2f}, 遷移信頼度: {transition_confidence:.2f}")
    #     
    #     printLog(f"\n==== 結果 ====")
    #     printLog(f"検出されたスライド遷移: {transition_count}件")
    #     printLog(f"ログファイル: inference.log")
    #     
    #     return True
    #     
    # except Exception as e:
    #     printLog(f"エラー: 結果保存に失敗しました: {e}")
    #     return False


def main():
    parser = argparse.ArgumentParser(description='SliTraNet 推論スクリプト')
    parser.add_argument('video_file', help='動画ファイルのパス')
    parser.add_argument('--roi-left-top', type=float, nargs=2, default=[0.23, 0.13],
                        help='ROI左上座標 (正規化座標 0.0-1.0) 例: --roi-left-top 0.23 0.13')
    parser.add_argument('--roi-right-bottom', type=float, nargs=2, default=[0.97, 0.88],
                        help='ROI右下座標 (正規化座標 0.0-1.0) 例: --roi-right-bottom 0.97 0.88')
    
    args = parser.parse_args()
    
    video_path = args.video_file
    roi_left_top = tuple(args.roi_left_top)
    roi_right_bottom = tuple(args.roi_right_bottom)
    
    # ROI座標の妥当性チェック
    if not (0.0 <= roi_left_top[0] <= 1.0 and 0.0 <= roi_left_top[1] <= 1.0):
        print("エラー: ROI左上座標は0.0-1.0の範囲で指定してください")
        sys.exit(1)
    if not (0.0 <= roi_right_bottom[0] <= 1.0 and 0.0 <= roi_right_bottom[1] <= 1.0):
        print("エラー: ROI右下座標は0.0-1.0の範囲で指定してください")
        sys.exit(1)
    if roi_left_top[0] >= roi_right_bottom[0] or roi_left_top[1] >= roi_right_bottom[1]:
        print("エラー: ROI座標が不正です（左上 < 右下である必要があります）")
        sys.exit(1)
    
    # ログファイル初期化
    if os.path.exists('inference.log'):
        os.remove('inference.log')
    
    printLog("=== SliTraNet 推論開始 ===")
    printLog(f"ROI設定: 左上{roi_left_top} 右下{roi_right_bottom}")
    
    # 推論実行
    success = run_inference(video_path, roi_left_top, roi_right_bottom)
    
    if success:
        printLog("=== 推論完了 ===")
        sys.exit(0)
    else:
        printLog("=== 推論失敗 ===")
        sys.exit(1)


if __name__ == '__main__':
    main()