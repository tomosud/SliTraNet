#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
動画から中間フレームを抽出するスクリプト
transitions.txtの各行間の中間フレームを抽出してPNG画像として保存
"""

import os
import sys
import csv
import subprocess
from datetime import datetime


def parse_transitions_file(transitions_file):
    """transitions.txtファイルを解析してフレーム番号のリストを返す"""
    frame_data = []
    
    try:
        with open(transitions_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)  # ヘッダー行をスキップ
            
            for row in reader:
                if len(row) >= 3:
                    transition_no = int(row[0])
                    frame_id0 = int(row[1])
                    frame_id1 = int(row[2])
                    frame_data.append((transition_no, frame_id0, frame_id1))
                    
    except Exception as e:
        print(f"Error parsing transitions file: {e}")
        return []
    
    return frame_data


def calculate_middle_frames(frame_data):
    """各行間の中間フレーム番号を計算"""
    middle_frames = []
    
    for i in range(len(frame_data) - 1):
        current_row = frame_data[i]
        next_row = frame_data[i + 1]
        
        # 現在の行のFrameID1と次の行のFrameID0の中間を計算
        frame1 = current_row[2]  # FrameID1
        frame2 = next_row[1]     # 次の行のFrameID0
        
        middle_frame = int((frame1 + frame2) / 2)
        middle_frames.append((i + 1, frame1, frame2, middle_frame))
        
        print(f"Between transition {current_row[0]} and {next_row[0]}: "
              f"frames {frame1}-{frame2}, middle: {middle_frame}")
    
    return middle_frames


def extract_frame_with_ffmpeg(video_file, frame_number, output_file):
    """ffmpegを使って指定フレームを画像として抽出"""
    cmd = [
        'ffmpeg',
        '-i', video_file,
        '-vf', f'select=eq(n\\,{frame_number})',
        '-vframes', '1',
        '-y',  # 上書き確認なし
        output_file
    ]
    
    try:
        # エンコーディングを指定してffmpegを実行
        result = subprocess.run(cmd, capture_output=True, text=True, 
                              encoding='utf-8', errors='replace')
        if result.returncode == 0:
            print(f"Successfully extracted frame {frame_number} to {output_file}")
            return True
        else:
            print(f"Error extracting frame {frame_number}: {result.stderr}")
            return False
    except Exception as e:
        print(f"Exception during ffmpeg execution: {e}")
        return False


def main():
    if len(sys.argv) != 3:
        print("Usage: python extract_frames.py <transitions.txt> <video.mp4>")
        print("Or drag and drop both files onto extract_frames.bat")
        sys.exit(1)
    
    transitions_file = sys.argv[1]
    video_file = sys.argv[2]
    
    # ファイル存在確認
    if not os.path.exists(transitions_file):
        print(f"Error: Transitions file not found: {transitions_file}")
        sys.exit(1)
    
    if not os.path.exists(video_file):
        print(f"Error: Video file not found: {video_file}")
        sys.exit(1)
    
    print(f"Processing transitions file: {transitions_file}")
    print(f"Processing video file: {video_file}")
    
    # transitions.txtを解析
    frame_data = parse_transitions_file(transitions_file)
    if not frame_data:
        print("Error: No valid frame data found in transitions file")
        sys.exit(1)
    
    print(f"Found {len(frame_data)} transitions")
    
    # 中間フレームを計算
    middle_frames = calculate_middle_frames(frame_data)
    if not middle_frames:
        print("No middle frames to extract")
        sys.exit(0)
    
    print(f"Will extract {len(middle_frames)} middle frames")
    
    # 出力ディレクトリ（transitions.txtと同じフォルダ）
    output_dir = os.path.dirname(transitions_file)
    
    # タイムスタンプ
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 各中間フレームを抽出
    success_count = 0
    for i, (transition_idx, frame1, frame2, middle_frame) in enumerate(middle_frames):
        output_filename = f"frame_{middle_frame:06d}_{timestamp}.png"
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"\n[{i+1}/{len(middle_frames)}] Extracting frame {middle_frame}")
        print(f"  Between frames {frame1} and {frame2}")
        print(f"  Output: {output_filename}")
        
        if extract_frame_with_ffmpeg(video_file, middle_frame, output_path):
            success_count += 1
        else:
            print(f"  Failed to extract frame {middle_frame}")
    
    print(f"\nCompleted: {success_count}/{len(middle_frames)} frames extracted successfully")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()