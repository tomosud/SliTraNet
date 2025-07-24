# -*- coding: utf-8 -*-
"""
SliTraNet共通ユーティリティ関数
"""

import os
import sys
import logging
import subprocess
import json
from pathlib import Path


def setup_logging(log_file='inference.log'):
    """ログ設定の初期化"""
    # 既存のログファイルを削除
    if os.path.exists(log_file):
        os.remove(log_file)
    
    # ログ設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def printLog(*args, **kwargs):
    """ログ出力用関数（後方互換性のため）"""
    print(*args, **kwargs)
    with open('inference.log', 'a', encoding='utf-8') as file:
        print(*args, **kwargs, file=file)


def is_video_file(file_path):
    """動画ファイルかどうかを判定"""
    if not os.path.exists(file_path):
        return False
    
    video_extensions = {'.mp4', '.avi', '.mov', '.m4v', '.mkv', '.wmv', '.flv', '.webm'}
    file_ext = Path(file_path).suffix.lower()
    return file_ext in video_extensions


def validate_video_file(video_file):
    """動画ファイルの妥当性検証"""
    if not os.path.exists(video_file):
        return False
    
    if not is_video_file(video_file):
        return False
    
    # ファイルサイズチェック（0バイトでないこと）
    if os.path.getsize(video_file) == 0:
        return False
    
    return True


def validate_roi_coordinates(roi_left_top, roi_right_bottom):
    """ROI座標の妥当性検証"""
    try:
        x1, y1 = roi_left_top
        x2, y2 = roi_right_bottom
        
        # 範囲チェック (0.0-1.0)
        if not (0.0 <= x1 <= 1.0 and 0.0 <= y1 <= 1.0):
            return False
        
        if not (0.0 <= x2 <= 1.0 and 0.0 <= y2 <= 1.0):
            return False
        
        # 左上 < 右下 のチェック
        if x1 >= x2 or y1 >= y2:
            return False
        
        return True
        
    except (ValueError, TypeError):
        return False


def get_video_info_ffmpeg(video_path):
    """ffmpegを使用して動画情報を取得"""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams',
            video_path
        ]
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
                    'width': stream['width'],
                    'height': stream['height'],
                    'fps': fps,
                    'total_frames': total_frames,
                    'duration': float(stream.get('duration', 0))
                }
        
        raise ValueError("Video stream not found")
        
    except Exception as e:
        # フォールバック: decordを使用
        try:
            import decord
            from decord import VideoReader
            
            vr = VideoReader(video_path)
            frame = vr[0]
            H, W, _ = frame.shape
            
            return {
                'width': W,
                'height': H,
                'fps': 30.0,  # デフォルト値
                'total_frames': len(vr),
                'duration': len(vr) / 30.0
            }
        except:
            return {
                'width': 0,
                'height': 0,
                'fps': 30.0,
                'total_frames': 0,
                'duration': 0
            }


def frame_to_timestamp(frame_number, fps):
    """フレーム番号をタイムスタンプ（HH:MM:SS.mmm）に変換"""
    seconds = frame_number / fps
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}h{minutes:02d}m{seconds:06.3f}s"


def ensure_directory(dir_path):
    """ディレクトリの存在を確認し、必要に応じて作成"""
    try:
        os.makedirs(dir_path, exist_ok=True)
        return True
    except Exception:
        return False


def safe_file_operation(func, *args, **kwargs):
    """ファイル操作を安全に実行"""
    try:
        return func(*args, **kwargs), None
    except Exception as e:
        return None, str(e)


def check_dependencies():
    """依存関係のチェック"""
    missing_deps = []
    
    # Python モジュール
    required_modules = [
        'torch', 'torchvision', 'cv2', 'numpy', 'decord'
    ]
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_deps.append(f"Python module: {module}")
    
    # 外部コマンド
    external_commands = ['ffmpeg', 'ffprobe']
    
    for cmd in external_commands:
        try:
            subprocess.run([cmd, '-version'], 
                         capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            missing_deps.append(f"External command: {cmd}")
    
    return missing_deps


def get_project_root():
    """プロジェクトのルートディレクトリを取得"""
    return Path(__file__).parent.absolute()


def get_model_paths():
    """モデルファイルのパスを取得"""
    root = get_project_root()
    weights_dir = root / "weights"
    
    return {
        'frame_similarity_2d': weights_dir / "Frame_similarity_ResNet18_gray.pth",
        'slide_video_detection_3d': weights_dir / "Slide_video_detection_3DResNet50.pth",
        'slide_transition_detection_3d': weights_dir / "Slide_transition_detection_3DResNet50.pth"
    }


def check_model_files():
    """モデルファイルの存在確認"""
    model_paths = get_model_paths()
    missing_models = []
    
    for model_name, model_path in model_paths.items():
        if not model_path.exists():
            missing_models.append(str(model_path))
    
    return missing_models