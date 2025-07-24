#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
画像重複検出プロジェクト - SliTraNet統合版
imagededupライブラリを使用して講演・プレゼンテーションのスクリーンショット画像から
相似性の高い画像を検出し、整理するツール

SliTraNetとの統合により、動画からスライド抽出後の重複除去を自動化
"""

import os
import sys
import glob
import datetime
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm

from imagededup.methods import DHash


class ImageSimilarityAnalyzer:
    """画像相似性分析クラス - SliTraNet統合版"""
    
    def __init__(self, target_dir: str, threshold: float = 0.85):
        """
        初期化
        
        Args:
            target_dir: extracted_framesフォルダのパス
            threshold: 相似性の閾値（0.0-1.0）
        """
        self.target_dir = Path(target_dir)
        self.dupp_dir = self.target_dir / "dupp"
        self.threshold = threshold
        self.dhasher = DHash()
        
        # duppディレクトリの作成
        self.dupp_dir.mkdir(exist_ok=True)
    
    def get_image_files(self) -> List[str]:
        """
        対象フォルダからPNGファイルを取得し、ファイル名順にソート（サブフォルダは含まない）
        
        Returns:
            ソート済みの画像ファイルパスのリスト
        """
        if not self.target_dir.exists():
            raise FileNotFoundError(f"対象フォルダが見つかりません: {self.target_dir}")
        
        # PNG画像を取得してソート（サブフォルダは含まない）
        png_files = [f for f in self.target_dir.glob("*.png") if f.is_file()]
        png_files.sort(key=lambda x: x.name)
        
        if not png_files:
            raise FileNotFoundError(f"PNG画像が見つかりません: {self.target_dir}")
        
        print(f"対象画像数: {len(png_files)}")
        return [str(f) for f in png_files]
    
    def calculate_image_hashes(self, image_files: List[str]) -> Dict[str, str]:
        """
        画像のハッシュ値を計算
        
        Args:
            image_files: 画像ファイルパスのリスト
            
        Returns:
            ファイル名をキー、ハッシュ値を値とする辞書
        """
        print("画像ハッシュを計算中...")
        
        # imagededupは画像ディレクトリを直接処理するため、
        # 対象ディレクトリのパスを渡す
        encodings = self.dhasher.encode_images(image_dir=str(self.target_dir))
        
        return encodings
    
    def find_duplicates(self, encodings: Dict[str, str]) -> Dict[str, List[str]]:
        """
        重複画像を検出
        
        Args:
            encodings: 画像ハッシュの辞書
            
        Returns:
            重複画像の辞書（キー画像：類似画像リスト）
        """
        print("重複画像を検出中...")
        
        # dHashの閾値を調整（imagededupでは0-10の範囲で指定、低いほど厳しい）
        # 0.85の類似度は、10-8.5=1.5程度の差分を許容（dHashはより敏感なため少し緩めに）
        max_distance_threshold = int((1.0 - self.threshold) * 10)
        
        duplicates = self.dhasher.find_duplicates(
            encoding_map=encodings,
            max_distance_threshold=max_distance_threshold
        )
        
        return duplicates
    
    def analyze_similarity_groups(self, image_files: List[str], 
                                 duplicates: Dict[str, List[str]]) -> List[Tuple[int, int]]:
        """
        相似画像の連続範囲を分析してグループ化（dHash向けに最適化）
        
        Args:
            image_files: ソート済み画像ファイルパスのリスト
            duplicates: 重複画像の辞書
            
        Returns:
            (開始インデックス, 終了インデックス)のタプルのリスト
        """
        print("相似グループを分析中...")
        
        # ファイル名からインデックスへのマッピングを作成
        file_to_index = {}
        for i, file_path in enumerate(image_files):
            filename = Path(file_path).name
            file_to_index[filename] = i
        
        # 全ての相似関係を記録
        similarity_graph = {}
        for i in range(len(image_files)):
            similarity_graph[i] = set()
        
        # 重複辞書から相似関係を構築
        for main_image, similar_images in duplicates.items():
            if main_image in file_to_index and similar_images:
                main_idx = file_to_index[main_image]
                for similar_image in similar_images:
                    if similar_image in file_to_index:
                        similar_idx = file_to_index[similar_image]
                        # 双方向の関係を記録
                        similarity_graph[main_idx].add(similar_idx)
                        similarity_graph[similar_idx].add(main_idx)
        
        # Union-Findを使用してグループを検出
        parent = list(range(len(image_files)))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # 相似関係のある画像をグループ化
        for i, similar_set in similarity_graph.items():
            for j in similar_set:
                union(i, j)
        
        # グループごとに画像をまとめる
        groups_dict = {}
        for i in range(len(image_files)):
            root = find(i)
            if root not in groups_dict:
                groups_dict[root] = []
            groups_dict[root].append(i)
        
        # 連続する範囲のみを抽出してグループ化
        final_groups = []
        for group_indices in groups_dict.values():
            group_indices.sort()
            
            # 連続する範囲に分割
            if len(group_indices) == 1:
                final_groups.append((group_indices[0], group_indices[0]))
            else:
                # 連続する部分を検出
                start = group_indices[0]
                prev = start
                
                for i in range(1, len(group_indices)):
                    current = group_indices[i]
                    if current != prev + 1:
                        # 連続が切れた場合、前の範囲を記録
                        final_groups.append((start, prev))
                        start = current
                    prev = current
                
                # 最後の範囲を記録
                final_groups.append((start, prev))
        
        # インデックス順にソート
        final_groups.sort()
        
        return final_groups
    
    def format_groups_output(self, groups: List[Tuple[int, int]]) -> List[str]:
        """
        グループを出力フォーマットに変換
        
        Args:
            groups: (開始インデックス, 終了インデックス)のタプルのリスト
            
        Returns:
            フォーマット済みの文字列のリスト
        """
        formatted = []
        for start, end in groups:
            if start == end:
                # 単体画像
                formatted.append(str(start))
            else:
                # 範囲画像
                formatted.append(f"{start},{end}")
        
        return formatted
    
    def save_results(self, image_files: List[str], groups: List[Tuple[int, int]]):
        """
        分析結果をファイルに保存し、相似画像をduppフォルダに移動
        
        Args:
            image_files: 画像ファイルパスのリスト
            groups: 相似グループのリスト
        """
        # similarity_groups.txtを動画と同じフォルダに保存（SliTraNet統合仕様）
        video_dir = self.target_dir.parent
        output_file = video_dir / "similarity_groups.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            # ヘッダー情報
            f.write("# 画像相似性分析結果 - SliTraNet統合版\n")
            f.write(f"# 処理日時: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# 対象画像数: {len(image_files)}\n")
            f.write(f"# 検出手法: DHash\n")
            f.write(f"# 類似度閾値: {self.threshold:.2f}\n")
            f.write("\n# 相似グループ（範囲表記）\n")
            
            # グループ情報
            formatted_groups = self.format_groups_output(groups)
            for group in formatted_groups:
                f.write(f"{group}\n")
        
        print(f"結果を保存しました: {output_file}")
        
        # 相似画像の移動処理（最初の1枚を残して他をduppフォルダに移動）
        self.move_duplicate_images(image_files, groups)
        
        # 統計情報の表示
        single_images = sum(1 for start, end in groups if start == end)
        group_images = len(groups) - single_images
        total_similar_images = sum(end - start + 1 for start, end in groups if start != end)
        moved_images = sum(end - start for start, end in groups if start != end)
        
        print(f"\n=== 重複画像除去結果サマリー ===")
        print(f"総画像数: {len(image_files)}")
        print(f"相似グループ数: {group_images}")
        print(f"単体画像数: {single_images}")
        print(f"相似画像の総数: {total_similar_images}")
        print(f"duppフォルダに移動した画像数: {moved_images}")
        print(f"最終画像数: {len(image_files) - moved_images}")
    
    def move_duplicate_images(self, image_files: List[str], groups: List[Tuple[int, int]]):
        """
        相似画像の移動処理：最初の1枚を残して他をduppフォルダに移動
        
        Args:
            image_files: 画像ファイルパスのリスト
            groups: 相似グループのリスト
        """
        print("\n相似画像をduppフォルダに移動中...")
        
        moved_count = 0
        
        for start, end in tqdm(groups, desc="画像を移動中"):
            if start != end:  # 相似グループの場合のみ処理
                # 最初の1枚(start)は残して、それ以外(start+1からend)をduppフォルダに移動
                for i in range(start + 1, end + 1):
                    source_path = Path(image_files[i])
                    dest_path = self.dupp_dir / source_path.name
                    
                    # 同名ファイルが存在する場合はリネーム
                    if dest_path.exists():
                        base_name = dest_path.stem
                        extension = dest_path.suffix
                        counter = 1
                        while dest_path.exists():
                            dest_path = self.dupp_dir / f"{base_name}_{counter}{extension}"
                            counter += 1
                    
                    shutil.move(str(source_path), str(dest_path))
                    moved_count += 1
        
        print(f"\n画像の移動が完了しました:")
        print(f"- 移動した画像数: {moved_count}")
        print(f"- 移動先: {self.dupp_dir}")
    
    def run_analysis(self):
        """
        画像相似性分析の実行 - SliTraNet統合版
        """
        try:
            print("\n=== 重複画像検出処理 ===")
            print(f"類似度閾値: {self.threshold}")
            print()
            
            # 1. 画像ファイルの取得
            image_files = self.get_image_files()
            
            # 2. ハッシュ値の計算
            encodings = self.calculate_image_hashes(image_files)
            
            # 3. 重複画像の検出
            duplicates = self.find_duplicates(encodings)
            
            # 4. 相似グループの分析
            groups = self.analyze_similarity_groups(image_files, duplicates)
            
            # 5. 結果の保存
            self.save_results(image_files, groups)
            
            print("\n重複画像除去処理が正常に完了しました！")
            
        except Exception as e:
            print(f"重複画像除去処理でエラーが発生しました: {e}")
            raise


def run_duplicate_removal(extracted_frames_dir: str, threshold: float = 0.85):
    """
    SliTraNet統合用の重複除去実行関数
    
    Args:
        extracted_frames_dir: extracted_framesフォルダのパス
        threshold: 類似度閾値（デフォルト85%）
    """
    if not os.path.exists(extracted_frames_dir):
        print(f"警告: extracted_framesフォルダが見つかりません: {extracted_frames_dir}")
        return
    
    # PNG画像の存在確認
    png_files = [f for f in os.listdir(extracted_frames_dir) 
                 if f.lower().endswith('.png')]
    
    if not png_files:
        print(f"警告: extracted_framesフォルダにPNG画像がありません: {extracted_frames_dir}")
        return
    
    if len(png_files) < 2:
        print(f"情報: 画像が{len(png_files)}枚のため、重複除去をスキップします")
        return
    
    print(f"extracted_framesフォルダ: {extracted_frames_dir}")
    print(f"検出画像数: {len(png_files)}枚")
    
    # 分析器を初期化して実行
    analyzer = ImageSimilarityAnalyzer(target_dir=extracted_frames_dir, threshold=threshold)
    analyzer.run_analysis()