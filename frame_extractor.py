# -*- coding: utf-8 -*-
"""
SliTraNetフレーム抽出処理
検出されたスライド遷移から中間フレームを抽出
"""

import os
import sys
import csv
import logging
import numpy as np
import cv2
from pathlib import Path

from utils import get_video_info_ffmpeg, frame_to_timestamp, printLog, ensure_directory

try:
    from decord import VideoReader, cpu, gpu
    import torch
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False


def parse_results_file(results_file):
    """_results.txtファイルを解析してスライド情報を返す"""
    logger = logging.getLogger(__name__)
    frame_data = []
    
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)  # ヘッダー行をスキップ
            
            for row in reader:
                if len(row) >= 3:
                    slide_no = int(row[0])
                    frame_id0 = int(row[1])
                    frame_id1 = int(row[2])
                    
                    # Slide No が -1 の行は無視（動画区間）
                    if slide_no == -1:
                        printLog(f"動画区間をスキップ: Slide {slide_no}, frames {frame_id0}-{frame_id1}")
                        continue
                    
                    # フレーム差が30未満の場合はスキップ
                    frame_diff = frame_id1 - frame_id0
                    if frame_diff < 30:
                        printLog(f"短すぎるスライドをスキップ: Slide {slide_no}, frames {frame_id0}-{frame_id1} (diff: {frame_diff})")
                        continue
                    
                    printLog(f"有効なスライド: Slide {slide_no}, frames {frame_id0}-{frame_id1} (diff: {frame_diff})")
                    frame_data.append((slide_no, frame_id0, frame_id1))
                    
        logger.info(f"有効なスライド数: {len(frame_data)}")
        return frame_data
                    
    except Exception as e:
        error_msg = f"結果ファイル解析エラー: {e}"
        printLog(error_msg)
        logger.error(error_msg)
        return []


def calculate_middle_frames(frame_data):
    """各スライドの中間フレームを計算"""
    logger = logging.getLogger(__name__)
    middle_frames = []
    
    for slide_no, frame_id0, frame_id1 in frame_data:
        # 中間フレームを計算
        middle_frame = int((frame_id0 + frame_id1) / 2)
        frame_diff = frame_id1 - frame_id0
        
        middle_frames.append({
            'slide_no': slide_no,
            'start_frame': frame_id0,
            'end_frame': frame_id1,
            'middle_frame': middle_frame,
            'frame_diff': frame_diff
        })
        
        printLog(f"Slide {slide_no}: frames {frame_id0}-{frame_id1} (diff: {frame_diff}), middle: {middle_frame}")
    
    logger.info(f"中間フレーム計算完了: {len(middle_frames)}件")
    return middle_frames


def save_image_japanese_path(image, filepath, log_func=printLog):
    """
    日本語パス対応の画像保存関数
    cv2.imwrite()が日本語パスで失敗する問題を解決
    """
    try:
        # 方法1: cv2.imwrite()を直接試行
        success = cv2.imwrite(filepath, image)
        if success:
            return True
        else:
            log_func(f"⚠ cv2.imwrite()失敗、代替方法を試行中...")
    except Exception as e:
        log_func(f"⚠ cv2.imwrite()例外: {e}, 代替方法を試行中...")
    
    try:
        # 方法2: cv2.imencode() + numpy.tofile()
        success, encoded_img = cv2.imencode('.png', image)
        if success:
            encoded_img.tofile(filepath)
            return True
        else:
            log_func(f"✗ cv2.imencode()失敗")
            return False
    except Exception as e:
        log_func(f"✗ 代替保存方法でも失敗: {e}")
        return False


def get_video_info_with_decord(video_path, log_func=printLog):
    """decordを使用した動画情報取得（test_extract.pyと同じ実装）"""
    vr = None
    ctx_type = "Unknown"
    
    # まずGPUを試行
    if torch.cuda.is_available():
        try:
            ctx = gpu(0)
            vr = VideoReader(video_path, ctx=ctx)
            ctx_type = "GPU"
            log_func(f"使用デバイス: GPU (成功)")
        except Exception as gpu_error:
            log_func(f"GPU初期化失敗: {gpu_error}")
            log_func("CPUにフォールバック中...")
            vr = None
    
    # GPUが失敗した場合、またはCUDAが利用できない場合はCPUを使用
    if vr is None:
        try:
            ctx = cpu(0)
            vr = VideoReader(video_path, ctx=ctx)
            ctx_type = "CPU"
            log_func(f"使用デバイス: CPU")
        except Exception as cpu_error:
            log_func(f"✗ CPU初期化も失敗: {cpu_error}")
            return None
    
    try:
        total_frames = len(vr)
        fps = vr.get_avg_fps()
        duration = total_frames / fps
        
        log_func(f"動画情報取得成功:")
        log_func(f"  総フレーム数: {total_frames:,}")
        log_func(f"  FPS: {fps:.2f}")
        log_func(f"  再生時間: {duration:.2f}秒 ({duration/60:.1f}分)")
        
        return {
            'vr': vr,
            'total_frames': total_frames,
            'fps': fps,
            'duration': duration,
            'device': ctx_type
        }
    except Exception as e:
        log_func(f"✗ 動画情報取得エラー: {e}")
        return None


def extract_frames_with_decord(video_file, middle_frames, output_dir, fps):
    """decordを使用した高速フレーム抽出（test_extract.pyと同じ実装ベース）"""
    logger = logging.getLogger(__name__)
    
    if not DECORD_AVAILABLE:
        raise Exception("decordが利用できません。requirements.txtから依存関係をインストールしてください。")
    
    # 動画情報取得（test_extract.pyと同じ方法）
    video_info = get_video_info_with_decord(video_file, printLog)
    if not video_info:
        raise Exception("動画情報の取得に失敗しました")
    
    vr = video_info['vr']
    device_type = video_info['device']
    logger.info(f"Using {device_type} for decord")
    
    try:
        # フレーム番号リストを準備
        frame_indices = [info['middle_frame'] for info in middle_frames]
        
        printLog(f"decordで {len(frame_indices)} フレームを一括抽出中...")
        
        # decordで一括フレーム抽出（複数のメソッドを試行）
        try:
            frames_tensor = vr.get_batch(frame_indices)
            
            # テンソル変換（複数のメソッドを試行）
            if hasattr(frames_tensor, 'asnumpy'):
                frames = frames_tensor.asnumpy()
                printLog(f"✓ asnumpy()で変換成功")
            elif hasattr(frames_tensor, 'numpy'):
                frames = frames_tensor.numpy()
                printLog(f"✓ numpy()で変換成功")
            else:
                frames = np.array(frames_tensor)
                printLog(f"✓ np.array()で変換成功")
            
            printLog(f"✓ decord一括取得完了")
            printLog(f"フレームデータ形状: {frames.shape}")
            printLog(f"データ型: {frames.dtype}")
        except Exception as e:
            printLog(f"✗ decordフレーム取得エラー: {e}")
            raise Exception(f"decordフレーム取得エラー: {e}")
        
        # 画像保存処理
        extracted_files = []
        success_count = 0
        fail_count = 0
        
        for i, (frame_index, frame) in enumerate(zip(frame_indices, frames)):
            slide_info = middle_frames[i]
            slide_no = slide_info['slide_no']
            
            # タイムスタンプ計算
            timestamp = frame_to_timestamp(frame_index, fps)
            
            # ファイル名作成（既存形式互換）
            filename = f"slide_{slide_no:03d}_frame_{frame_index:06d}_{timestamp}.png"
            filepath = os.path.join(output_dir, filename)
            
            # RGB→BGR変換
            try:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            except Exception as e:
                printLog(f"✗ 色空間変換エラー (slide {slide_no}): {e}")
                fail_count += 1
                continue
            
            # 日本語パス対応保存
            if save_image_japanese_path(frame_bgr, filepath):
                extracted_files.append(filepath)
                success_count += 1
                printLog(f"  ✓ Slide {slide_no}: frame {frame_index} -> {filename}")
            else:
                fail_count += 1
                error_msg = f"Slide {slide_no} frame {frame_index} 保存失敗: {filepath}"
                printLog(f"  ✗ {error_msg}")
                logger.error(error_msg)
        
        # 結果サマリー
        printLog(f"decord抽出完了: {success_count}/{len(middle_frames)} フレーム")
        printLog(f"成功率: {success_count/len(middle_frames)*100:.1f}%")
        logger.info(f"decord extraction completed: {success_count}/{len(middle_frames)} frames, success rate: {success_count/len(middle_frames)*100:.1f}%")
        
        return success_count
        
    except Exception as e:
        error_msg = f"decord抽出でエラー: {e}"
        printLog(f"✗ {error_msg}")
        logger.error(error_msg)  
        raise Exception(error_msg)


def extract_frames_batch(video_file, middle_frames, output_dir, fps):
    """decordを使用した高速フレーム抽出"""
    logger = logging.getLogger(__name__)
    
    printLog("高速抽出のためdecordを使用")
    logger.info("Using decord for high-speed frame extraction")
    
    return extract_frames_with_decord(video_file, middle_frames, output_dir, fps)


def extract_slide_frames(results_file, video_file):
    """スライドフレーム抽出のメイン処理"""
    logger = logging.getLogger(__name__)
    
    try:
        printLog(f"結果ファイル: {results_file}")
        printLog(f"動画ファイル: {video_file}")
        logger.info(f"フレーム抽出開始: {results_file} -> {video_file}")
        
        # ファイル存在確認
        if not os.path.exists(results_file):
            error_msg = f"結果ファイルが見つかりません: {results_file}"
            printLog(f"エラー: {error_msg}")
            logger.error(error_msg)
            return False, None, 0
        
        if not os.path.exists(video_file):
            error_msg = f"動画ファイルが見つかりません: {video_file}"
            printLog(f"エラー: {error_msg}")
            logger.error(error_msg)
            return False, None, 0
        
        # 動画情報を取得
        printLog("動画情報を取得中...")
        video_info = get_video_info_ffmpeg(video_file)
        fps = video_info['fps']
        
        printLog(f"動画FPS: {fps:.3f}")
        printLog(f"総フレーム数: {video_info['total_frames']}")
        printLog(f"再生時間: {video_info['duration']:.3f}s")
        logger.info(f"動画情報 - FPS: {fps:.3f}, フレーム数: {video_info['total_frames']}")
        
        # 結果ファイルを解析
        printLog("結果ファイルを解析中...")
        frame_data = parse_results_file(results_file)
        
        if not frame_data:
            error_msg = "有効なフレームデータが見つかりません（30フレーム以上の差がある有効なスライドなし）"
            printLog(f"エラー: {error_msg}")
            logger.error(error_msg)
            return False, None, 0
        
        printLog(f"有効なスライド数: {len(frame_data)} （30フレーム以上、動画区間除外済み）")
        
        # 中間フレームを計算
        middle_frames = calculate_middle_frames(frame_data)
        if not middle_frames:
            error_msg = "抽出する中間フレームがありません"
            printLog(f"エラー: {error_msg}")
            logger.error(error_msg)
            return False, None, 0
        
        printLog(f"抽出予定フレーム数: {len(middle_frames)} （スライドあたり1フレーム）")
        
        # 出力ディレクトリ設定
        base_output_dir = os.path.dirname(results_file)
        output_dir = os.path.join(base_output_dir, "extracted_frames")
        
        if not ensure_directory(output_dir):
            error_msg = f"出力ディレクトリの作成に失敗: {output_dir}"
            printLog(f"エラー: {error_msg}")
            logger.error(error_msg)
            return False, None, 0
        
        printLog(f"出力ディレクトリ: {output_dir}")
        logger.info(f"出力ディレクトリ: {output_dir}")
        
        # 高速フレーム抽出（decord）
        printLog("=== フレーム抽出開始 ===")
        total_success = extract_frames_batch(video_file, middle_frames, output_dir, fps)
        
        printLog("=== フレーム抽出完了 ===")
        printLog(f"抽出成功: {total_success}/{len(middle_frames)} フレーム")
        printLog(f"成功率: {total_success/len(middle_frames)*100:.1f}%")
        printLog(f"出力ディレクトリ: {output_dir}")
        
        logger.info(f"フレーム抽出完了: {total_success}/{len(middle_frames)} 成功")
        
        return True, output_dir, total_success
        
    except Exception as e:
        error_msg = f"フレーム抽出処理でエラーが発生: {e}"
        printLog(f"エラー: {error_msg}")
        logger.error(error_msg, exc_info=True)
        return False, None, 0