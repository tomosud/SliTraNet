#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
フレーム抽出テストツール
decord動作検証用 - 日本語パス対応版

目的: decordフレーム抽出機能の独立検証とデバッグ
- 動画全尺を等間隔で100分割して100枚のフレームを抽出
- 日本語パス対応の画像保存機能を実装
- 詳細なエラーログ出力
"""

import sys
import os
import time
from pathlib import Path
import numpy as np
import cv2
import torch
from decord import VideoReader, cpu, gpu
from datetime import datetime, timedelta


def setup_logging(output_dir):
    """ログファイルの設定"""
    log_file = os.path.join(output_dir, "extraction_log.txt")
    
    def log_message(message, also_print=True):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        
        if also_print:
            print(message)
        
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(log_entry + "\n")
        except Exception as e:
            print(f"ログ書き込みエラー: {e}")
    
    return log_message


def save_image_japanese_path(image, filepath, log_func):
    """
    日本語パス対応の画像保存関数
    cv2.imwrite()が日本語パスで失敗する問題を解決
    """
    try:
        # 方法1: cv2.imwrite()を直接試行
        success = cv2.imwrite(filepath, image)
        if success:
            log_func(f"✓ cv2.imwrite()で保存成功: {filepath}", False)
            return True
        else:
            log_func(f"⚠ cv2.imwrite()失敗、代替方法を試行中...", False)
    except Exception as e:
        log_func(f"⚠ cv2.imwrite()例外: {e}, 代替方法を試行中...", False)
    
    try:
        # 方法2: cv2.imencode() + numpy.tofile()
        success, encoded_img = cv2.imencode('.png', image)
        if success:
            encoded_img.tofile(filepath)
            log_func(f"✓ 代替方法で保存成功: {filepath}", False)
            return True
        else:
            log_func(f"✗ cv2.imencode()失敗", False)
            return False
    except Exception as e:
        log_func(f"✗ 代替保存方法でも失敗: {e}", False)
        return False


def get_video_info(video_path, log_func):
    """動画情報を取得"""
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


def extract_test_frames(video_path, output_dir, log_func):
    """テスト用フレーム抽出メイン処理"""
    
    # 動画情報取得
    video_info = get_video_info(video_path, log_func)
    if not video_info:
        return False
    
    vr = video_info['vr']
    total_frames = video_info['total_frames']
    fps = video_info['fps']
    
    # 100等分のフレーム番号リスト生成
    num_test_frames = 100
    frame_indices = []
    
    for i in range(num_test_frames):
        frame_index = int((i * total_frames) / num_test_frames)
        # 範囲チェック
        if frame_index >= total_frames:
            frame_index = total_frames - 1
        frame_indices.append(frame_index)
    
    log_func(f"抽出対象フレーム: {len(frame_indices)}枚")
    log_func(f"フレーム番号範囲: {frame_indices[0]} ～ {frame_indices[-1]}")
    
    # decordで一括フレーム抽出
    log_func("decordフレーム一括取得開始...")
    start_time = time.time()
    
    try:
        frames = vr.get_batch(frame_indices).asnumpy()
        extraction_time = time.time() - start_time
        log_func(f"✓ decord一括取得完了: {extraction_time:.2f}秒")
        log_func(f"フレームデータ形状: {frames.shape}")
        log_func(f"データ型: {frames.dtype}")
    except Exception as e:
        log_func(f"✗ decordフレーム取得エラー: {e}")
        return False
    
    # 画像保存処理
    log_func("画像保存処理開始...")
    save_start_time = time.time()
    
    success_count = 0
    fail_count = 0
    
    for i, (frame_index, frame) in enumerate(zip(frame_indices, frames)):
        # タイムスタンプ計算
        timestamp_sec = frame_index / fps
        hours = int(timestamp_sec // 3600)
        minutes = int((timestamp_sec % 3600) // 60)
        seconds = timestamp_sec % 60
        timestamp_str = f"{hours:02d}h{minutes:02d}m{seconds:06.3f}s"
        
        # ファイル名生成
        filename = f"test_frame_{i+1:03d}_frame_{frame_index:06d}_{timestamp_str}.png"
        filepath = os.path.join(output_dir, filename)
        
        # RGB→BGR変換
        try:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        except Exception as e:
            log_func(f"✗ 色空間変換エラー (frame {i+1}): {e}")
            fail_count += 1
            continue
        
        # 画像保存
        if save_image_japanese_path(frame_bgr, filepath, log_func):
            success_count += 1
            if (i + 1) % 10 == 0:  # 10枚ごとに進捗表示
                log_func(f"進捗: {i+1}/{len(frame_indices)} ({success_count}成功, {fail_count}失敗)")
        else:
            fail_count += 1
    
    save_time = time.time() - save_start_time
    
    # 結果サマリー
    log_func("=" * 50)
    log_func("フレーム抽出テスト結果:")
    log_func(f"  対象フレーム数: {len(frame_indices)}")
    log_func(f"  成功: {success_count}")
    log_func(f"  失敗: {fail_count}")
    log_func(f"  成功率: {success_count/len(frame_indices)*100:.1f}%")
    log_func(f"  decord処理時間: {extraction_time:.2f}秒")
    log_func(f"  画像保存時間: {save_time:.2f}秒")
    log_func(f"  総処理時間: {extraction_time + save_time:.2f}秒")
    
    # ファイル存在確認
    log_func("ファイル存在確認...")
    actual_files = []
    for filename in os.listdir(output_dir):
        if filename.startswith("test_frame_") and filename.endswith(".png"):
            filepath = os.path.join(output_dir, filename)
            if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                actual_files.append(filename)
    
    log_func(f"実際に作成されたファイル数: {len(actual_files)}")
    
    if len(actual_files) != success_count:
        log_func(f"⚠ 注意: 成功カウント({success_count})と実ファイル数({len(actual_files)})が一致しません")
    
    log_func("=" * 50)
    
    return success_count > 0


def main():
    if len(sys.argv) != 2:
        print("使用方法: python test_extract.py <video_file>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    # 入力チェック
    if not os.path.exists(video_path):
        print(f"エラー: ファイルが見つかりません: {video_path}")
        sys.exit(1)
    
    # 出力ディレクトリ作成
    video_dir = os.path.dirname(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(video_dir, "test_extract")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # ログ設定
    log_func = setup_logging(output_dir)
    
    print("=" * 60)
    print("  フレーム抽出テストツール (decord検証用)")
    print("=" * 60)
    print(f"動画ファイル: {video_path}")
    print(f"出力先: {output_dir}")
    print()
    
    log_func("=" * 60)
    log_func("フレーム抽出テスト開始")
    log_func(f"動画ファイル: {video_path}")
    log_func(f"出力ディレクトリ: {output_dir}")
    log_func("=" * 60)
    
    # フレーム抽出実行
    try:
        success = extract_test_frames(video_path, output_dir, log_func)
        
        if success:
            print("✓ フレーム抽出テスト完了")
            print(f"結果は以下のフォルダに保存されました:")
            print(f"  {output_dir}")
            log_func("✓ フレーム抽出テスト正常完了")
            sys.exit(0)
        else:
            print("✗ フレーム抽出テスト失敗")
            log_func("✗ フレーム抽出テスト失敗")
            sys.exit(1)
            
    except Exception as e:
        error_msg = f"予期しないエラー: {e}"
        print(f"✗ {error_msg}")
        log_func(f"✗ {error_msg}")
        import traceback
        log_func(f"スタックトレース:\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()