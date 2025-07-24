# -*- coding: utf-8 -*-
"""
SliTraNet統合処理スクリプト
動画ファイルから自動でスライド遷移検出とフレーム抽出を実行

使用方法: python main.py <video_file> [--roi-left-top x y] [--roi-right-bottom x y]
"""

import os
import sys
import argparse
import logging
from pathlib import Path

from utils import setup_logging, validate_video_file, validate_roi_coordinates
from inference_core import run_slide_detection
from frame_extractor import extract_slide_frames
from image_similarity import run_duplicate_removal


def setup_argument_parser():
    """コマンドライン引数の設定"""
    parser = argparse.ArgumentParser(
        description='SliTraNet統合処理 - 動画からスライド遷移検出とフレーム抽出',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python main.py video.mp4
  python main.py video.mp4 --roi-left-top 0.2 0.1 --roi-right-bottom 0.9 0.8
  
デフォルトROI: 左上(0.23, 0.13) 右下(0.97, 0.88)
ROI座標は正規化座標(0.0-1.0)で指定してください
        """
    )
    
    parser.add_argument('video_file', 
                       help='入力動画ファイルのパス')
    
    parser.add_argument('--roi-left-top', 
                       type=float, nargs=2, default=[0.23, 0.13],
                       metavar=('X', 'Y'),
                       help='ROI左上座標 (正規化座標 0.0-1.0) [デフォルト: 0.23 0.13]')
    
    parser.add_argument('--roi-right-bottom', 
                       type=float, nargs=2, default=[0.97, 0.88],
                       metavar=('X', 'Y'),
                       help='ROI右下座標 (正規化座標 0.0-1.0) [デフォルト: 0.97 0.88]')
    
    parser.add_argument('--debug', 
                       action='store_true',
                       help='デバッグモード（ROI可視化を有効化）')
    
    parser.add_argument('--keep-results', 
                       action='store_true',
                       help='中間ファイル(_results.txt)を保持する')
    
    return parser


def validate_arguments(args):
    """引数の妥当性検証"""
    try:
        # 動画ファイルの検証
        if not validate_video_file(args.video_file):
            return False, f"動画ファイルが無効です: {args.video_file}"
        
        # ROI座標の検証
        roi_left_top = tuple(args.roi_left_top)
        roi_right_bottom = tuple(args.roi_right_bottom)
        
        if not validate_roi_coordinates(roi_left_top, roi_right_bottom):
            return False, "ROI座標が無効です"
        
        return True, None
        
    except Exception as e:
        return False, f"引数検証エラー: {e}"


def print_processing_summary(video_file, roi_left_top, roi_right_bottom):
    """処理開始時のサマリー表示"""
    print("=" * 60)
    print("SliTraNet統合処理開始")
    print("=" * 60)
    print(f"入力動画: {video_file}")
    print(f"ROI設定: 左上{roi_left_top} 右下{roi_right_bottom}")
    print(f"処理手順: 1.スライド遷移検出 → 2.フレーム抽出 → 3.重複画像除去")
    print("=" * 60)


def print_completion_summary(success, video_file, results_file, extracted_frames_dir, total_frames=0):
    """処理完了時のサマリー表示"""
    print("\n" + "=" * 60)
    if success:
        print("✓ 統合処理が正常に完了しました")
        print("=" * 60)
        print(f"入力動画: {video_file}")
        print(f"検出結果: {results_file}")
        if total_frames > 0:
            print(f"抽出フレーム: {total_frames}枚")
            print(f"出力フォルダ: {extracted_frames_dir}")
            
            # 重複除去関連の出力情報
            video_dir = os.path.dirname(video_file)
            similarity_groups_file = os.path.join(video_dir, "similarity_groups.txt")
            dupp_dir = os.path.join(extracted_frames_dir, "dupp")
            
            if os.path.exists(similarity_groups_file):
                print(f"重複検出結果: {similarity_groups_file}")
            if os.path.exists(dupp_dir):
                dupp_count = len([f for f in os.listdir(dupp_dir) if f.lower().endswith('.png')])
                if dupp_count > 0:
                    print(f"重複画像({dupp_count}枚): {dupp_dir}")
        
        print("ログファイル: inference.log")
    else:
        print("✗ 統合処理中にエラーが発生しました")
        print("=" * 60)
        print("詳細はログファイル(inference.log)を確認してください")
    print("=" * 60)


def main():
    """メイン処理"""
    # 引数解析
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # 基本的な引数取得
    video_file = os.path.abspath(args.video_file)
    roi_left_top = tuple(args.roi_left_top)
    roi_right_bottom = tuple(args.roi_right_bottom)
    debug_mode = args.debug
    keep_results = args.keep_results
    
    # ログ設定
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # 引数検証
        valid, error_msg = validate_arguments(args)
        if not valid:
            print(f"エラー: {error_msg}")
            logger.error(f"引数検証失敗: {error_msg}")
            sys.exit(1)
        
        # 処理開始サマリー
        print_processing_summary(video_file, roi_left_top, roi_right_bottom)
        logger.info("統合処理開始")
        logger.info(f"動画ファイル: {video_file}")
        logger.info(f"ROI設定: 左上{roi_left_top} 右下{roi_right_bottom}")
        
        # ステップ1: スライド遷移検出
        print("\n[ステップ1/2] スライド遷移検出を実行中...")
        logger.info("Step 1: スライド遷移検出開始")
        
        detection_success, results_file = run_slide_detection(
            video_file, roi_left_top, roi_right_bottom, debug_mode
        )
        
        if not detection_success:
            error_msg = "スライド遷移検出に失敗しました"
            print(f"エラー: {error_msg}")
            logger.error(error_msg)
            print_completion_summary(False, video_file, None, None)
            sys.exit(1)
        
        print(f"✓ スライド遷移検出完了: {results_file}")
        logger.info(f"Step 1完了: {results_file}")
        
        # results_fileの存在確認
        if not os.path.exists(results_file):
            error_msg = f"検出結果ファイルが見つかりません: {results_file}"
            print(f"エラー: {error_msg}")
            logger.error(error_msg)
            print_completion_summary(False, video_file, results_file, None)
            sys.exit(1)
        
        # ステップ2: フレーム抽出
        print("\n[ステップ2/3] フレーム抽出を実行中...")
        logger.info("Step 2: フレーム抽出開始")
        
        extraction_success, extracted_frames_dir, total_frames = extract_slide_frames(
            results_file, video_file
        )
        
        if not extraction_success:
            error_msg = "フレーム抽出に失敗しました"
            print(f"エラー: {error_msg}")
            logger.error(error_msg)
            print_completion_summary(False, video_file, results_file, extracted_frames_dir)
            sys.exit(1)
        
        print(f"✓ フレーム抽出完了: {total_frames}枚抽出")
        logger.info(f"Step 2完了: {total_frames}枚抽出 -> {extracted_frames_dir}")
        
        # ステップ3: 重複画像除去
        print("\n[ステップ3/3] 重複画像除去を実行中...")
        logger.info("Step 3: 重複画像除去開始")
        
        try:
            # 重複画像除去を実行（デフォルト閾値85%）
            run_duplicate_removal(extracted_frames_dir, threshold=0.85)
            print("✓ 重複画像除去完了")
            logger.info("Step 3完了: 重複画像除去")
        except Exception as e:
            # 重複除去のエラーは警告扱い（処理は続行）
            error_msg = f"重複画像除去で警告: {e}"
            print(f"警告: {error_msg}")
            logger.warning(error_msg)
        
        # 中間ファイルのクリーンアップ（オプション）
        if not keep_results:
            logger.info("中間ファイルを保持します（--keep-resultsが有効）")
        else:
            logger.info("中間ファイルを保持します")
        
        # 成功時のサマリー表示
        print_completion_summary(True, video_file, results_file, extracted_frames_dir, total_frames)
        logger.info("統合処理が正常に完了しました")
        
        sys.exit(0)
        
    except KeyboardInterrupt:
        error_msg = "処理が中断されました"
        print(f"\n{error_msg}")
        logger.warning(error_msg)
        sys.exit(130)  # SIGINT exit code
        
    except Exception as e:
        error_msg = f"予期しないエラーが発生しました: {e}"
        print(f"エラー: {error_msg}")
        logger.error(error_msg, exc_info=True)
        print_completion_summary(False, video_file if 'video_file' in locals() else 'unknown', 
                                None, None)
        sys.exit(1)


if __name__ == '__main__':
    main()