# -*- coding: utf-8 -*-
"""
SliTraNetフレーム抽出処理
検出されたスライド遷移から中間フレームを抽出
"""

import os
import sys
import csv
import subprocess
import logging
from pathlib import Path

from utils import get_video_info_ffmpeg, frame_to_timestamp, printLog, ensure_directory


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


def extract_frames_batch(video_file, middle_frames, output_dir, fps, batch_size=10):
    """バッチ処理でフレームを高速抽出"""
    logger = logging.getLogger(__name__)
    total_success = 0
    total_batches = (len(middle_frames) + batch_size - 1) // batch_size
    
    for batch_idx in range(0, len(middle_frames), batch_size):
        batch = middle_frames[batch_idx:batch_idx+batch_size]
        batch_num = batch_idx // batch_size + 1
        
        printLog(f"バッチ {batch_num}/{total_batches} 処理中: フレーム {batch_idx+1}-{min(batch_idx+batch_size, len(middle_frames))}")
        
        # select式を構築（複数フレーム指定）
        select_parts = [f"eq(n\\,{info['middle_frame']})" for info in batch]
        select_expr = '+'.join(select_parts)
        
        # 一時ファイルパターン
        temp_pattern = os.path.join(output_dir, f"batch_{batch_num:03d}_frame_%03d.png")
        
        cmd = [
            'ffmpeg',
            '-i', video_file,
            '-vf', f"select='{select_expr}'",
            '-vsync', '0',  # フレーム番号維持
            '-y',  # 上書き確認なし
            temp_pattern
        ]
        
        try:
            # バッチ実行
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                  encoding='utf-8', errors='replace')
            
            if result.returncode == 0:
                # 一時ファイルをタイムスタンプ付きでリネーム
                batch_success = 0
                for i, slide_info in enumerate(batch):
                    temp_file = os.path.join(output_dir, f"batch_{batch_num:03d}_frame_{i+1:03d}.png")
                    
                    if os.path.exists(temp_file):
                        # タイムスタンプ付きファイル名を生成
                        slide_no = slide_info['slide_no']
                        middle_frame = slide_info['middle_frame']
                        timestamp = frame_to_timestamp(middle_frame, fps)
                        final_filename = f"slide_{slide_no:03d}_frame_{middle_frame:06d}_{timestamp}.png"
                        final_path = os.path.join(output_dir, final_filename)
                        
                        try:
                            os.rename(temp_file, final_path)
                            printLog(f"  ✓ Slide {slide_no}: frame {middle_frame} -> {final_filename}")
                            batch_success += 1
                        except Exception as e:
                            printLog(f"  ✗ リネームエラー {temp_file}: {e}")
                            # 一時ファイルを削除
                            try:
                                os.remove(temp_file)
                            except:
                                pass
                    else:
                        slide_no = slide_info['slide_no']
                        middle_frame = slide_info['middle_frame']
                        printLog(f"  ✗ Slide {slide_no}: frame {middle_frame} がバッチ出力で見つかりません")
                
                total_success += batch_success
                printLog(f"  バッチ {batch_num} 完了: {batch_success}/{len(batch)} フレーム抽出")
                logger.info(f"バッチ {batch_num} 完了: {batch_success}/{len(batch)}")
                
            else:
                error_msg = f"バッチ {batch_num} 失敗: {result.stderr}"
                printLog(f"  ✗ {error_msg}")
                logger.error(error_msg)
                
        except Exception as e:
            error_msg = f"バッチ {batch_num} 例外: {e}"
            printLog(f"  ✗ {error_msg}")
            logger.error(error_msg)
    
    logger.info(f"全バッチ処理完了: {total_success}/{len(middle_frames)} 成功")
    return total_success


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
        
        # バッチ処理でフレーム抽出
        printLog("=== バッチフレーム抽出開始 ===")
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