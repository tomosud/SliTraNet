#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
動画からフレーム範囲を抽出するスクリプト
長いスライド区間（30フレーム以上）の全フレームをタイムスタンプ付きで抽出
"""

import os
import sys
import csv
import subprocess
import json
import re
from datetime import datetime, timedelta


def get_video_info(video_file):
    """動画の情報（フレームレート、総フレーム数など）を取得"""
    cmd = [
        'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams',
        video_file
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        
        for stream in data['streams']:
            if stream['codec_type'] == 'video':
                # フレームレートを取得
                fps_str = stream.get('r_frame_rate', '30/1')
                fps_parts = fps_str.split('/')
                fps = float(fps_parts[0]) / float(fps_parts[1])
                
                # 総フレーム数を取得
                total_frames = int(stream.get('nb_frames', 0))
                
                return {
                    'fps': fps,
                    'total_frames': total_frames,
                    'duration': float(stream.get('duration', 0))
                }
        
        raise ValueError("Video stream not found")
    except Exception as e:
        print(f"Error getting video info: {e}")
        # デフォルト値を返す
        return {'fps': 30.0, 'total_frames': 0, 'duration': 0}


def frame_to_timestamp(frame_number, fps):
    """フレーム番号をタイムスタンプ（HH:MM:SS.mmm）に変換"""
    seconds = frame_number / fps
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}h{minutes:02d}m{seconds:06.3f}s"


def parse_transitions_file(transitions_file):
    """transitions.txtまたは_results.txtファイルを解析してフレーム番号のリストを返す"""
    frame_data = []
    
    try:
        with open(transitions_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)  # ヘッダー行をスキップ
            
            for row in reader:
                if len(row) >= 3:
                    slide_no = int(row[0])
                    frame_id0 = int(row[1])
                    frame_id1 = int(row[2])
                    
                    # Slide No が -1 の行は無視（動画区間）
                    if slide_no == -1:
                        print(f"Skipping video segment: Slide {slide_no}, frames {frame_id0}-{frame_id1}")
                        continue
                    
                    # フレーム差が30未満の場合はスキップ
                    frame_diff = frame_id1 - frame_id0
                    if frame_diff < 30:
                        print(f"Skipping short slide: Slide {slide_no}, frames {frame_id0}-{frame_id1} (diff: {frame_diff})")
                        continue
                    
                    print(f"Valid slide: Slide {slide_no}, frames {frame_id0}-{frame_id1} (diff: {frame_diff})")
                    frame_data.append((slide_no, frame_id0, frame_id1))
                    
    except Exception as e:
        print(f"Error parsing transitions file: {e}")
        return []
    
    return frame_data


def calculate_middle_frames(frame_data):
    """各スライドの中間フレームを計算"""
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
        
        print(f"Slide {slide_no}: frames {frame_id0}-{frame_id1} (diff: {frame_diff}), middle: {middle_frame}")
    
    return middle_frames


def extract_single_frame_with_ffmpeg(video_file, slide_info, output_dir, fps):
    """ffmpegを使って中間フレーム1枚を高速抽出"""
    slide_no = slide_info['slide_no']
    middle_frame = slide_info['middle_frame']
    start_frame = slide_info['start_frame']
    end_frame = slide_info['end_frame']
    
    print(f"Extracting slide {slide_no}: middle frame {middle_frame} (between {start_frame}-{end_frame})")
    
    # タイムスタンプを計算
    timestamp = frame_to_timestamp(middle_frame, fps)
    
    # 出力ファイル名（タイムスタンプ付き）
    output_filename = f"slide_{slide_no:03d}_frame_{middle_frame:06d}_{timestamp}.png"
    output_path = os.path.join(output_dir, output_filename)
    
    # 1フレームのみをselectフィルターで抽出（高速化）
    select_filter = f"select='eq(n,{middle_frame})'"
    
    cmd = [
        'ffmpeg',
        '-i', video_file,
        '-vf', select_filter,
        '-vframes', '1',  # 1フレームのみ
        '-y',  # 上書き確認なし
        output_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, 
                              encoding='utf-8', errors='replace')
        
        if result.returncode == 0:
            print(f"Successfully extracted frame {middle_frame} -> {output_filename}")
            return 1
        else:
            print(f"Error extracting frame {middle_frame}: {result.stderr}")
            return 0
            
    except Exception as e:
        print(f"Exception during ffmpeg execution for slide {slide_no}: {e}")
        return 0


def main():
    if len(sys.argv) != 3:
        print("Usage: python extract_frames.py <transitions.txt|results.txt> <video.mp4>")
        print("Extracts middle frame from long slides (30+ frame difference) with timestamps")
        print("Supports both _transitions.txt and _results.txt files")
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
    
    print(f"Processing file: {transitions_file}")
    print(f"Processing video file: {video_file}")
    
    # 動画情報を取得
    print("\nGetting video information...")
    video_info = get_video_info(video_file)
    fps = video_info['fps']
    print(f"Video FPS: {fps:.3f}")
    print(f"Total frames: {video_info['total_frames']}")
    print(f"Duration: {video_info['duration']:.3f}s")
    
    # ファイルタイプを判定
    if "transitions.txt" in transitions_file.lower():
        print("File type: Transitions file (_transitions.txt)")
    elif "results.txt" in transitions_file.lower():
        print("File type: Results file (_results.txt)")
    else:
        print("File type: Unknown (assuming same format)")
    
    # ファイルを解析
    print("\nParsing transitions file...")
    frame_data = parse_transitions_file(transitions_file)
    if not frame_data:
        print("Error: No valid frame data found (no slides with 30+ frame difference)")
        sys.exit(1)
    
    print(f"\nFound {len(frame_data)} valid slides (30+ frames, excluding video segments)")
    
    # 中間フレームを計算
    middle_frames = calculate_middle_frames(frame_data)
    if not middle_frames:
        print("No middle frames to extract")
        sys.exit(0)
    
    print(f"Will extract {len(middle_frames)} middle frames (1 per slide)")
    
    # 出力ディレクトリ（transitions.txtと同じフォルダにextracted_framesサブフォルダ）
    base_output_dir = os.path.dirname(transitions_file)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_output_dir, f"extracted_frames_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    
    # 各スライドの中間フレームを抽出
    total_success_count = 0
    for i, slide_info in enumerate(middle_frames):
        print(f"\n[{i+1}/{len(middle_frames)}] Processing slide {slide_info['slide_no']}")
        
        success_count = extract_single_frame_with_ffmpeg(video_file, slide_info, output_dir, fps)
        total_success_count += success_count
    
    print(f"\n=== Extraction Complete ===")
    print(f"Total frames extracted: {total_success_count}/{len(middle_frames)}")
    print(f"Success rate: {total_success_count/len(middle_frames)*100:.1f}%")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()