# SliTraNet 統合実装記録

## image_dupp統合計画 (2025-07-24)

### 統合目標
SliTraNetで動画からスライドフレームを抽出した後、image_duppツールを使用して類似画像を自動検出・削除し、最終的な画像数を最小化する一連の処理を実現する。

### 統合アーキテクチャ

#### 処理フロー
```
動画入力 → SliTraNetスライド抽出 → image_dupp重複除去 → 最終結果出力
    ↓              ↓                    ↓              ↓
 run.bat    extracted_frames/    similarity_groups.txt  final_slides/
```

#### ファイル構成計画
```
SliTraNet/
├── main.py                    # 統合エントリポイント（引数解析、フロー制御）
├── inference_core.py          # スライド遷移検出のコア処理
├── frame_extractor.py         # フレーム抽出処理
├── image_similarity.py        # NEW: image_duppから統合
├── utils.py                   # 共通ユーティリティ関数
├── run.bat                    # 統合バッチファイル（全処理実行）
├── setup.bat                  # 統合環境セットアップ
└── requirements.txt           # 統合依存関係
```

### 依存関係分析と統合

#### 現在の依存関係
**SliTraNet (requirements.txt)**:
- `torch` (CUDA 12.4+)
- `torchvision` (CUDA 12.4+)
- `opencv-contrib-python-headless`
- `numpy`
- `decord`

**image_dupp (requirements.txt)**:
- `imagededup`
- `opencv-python`
- `Pillow`
- `numpy`
- `tqdm`

#### 依存関係の競合と解決策

**⚠️ 主要な競合: OpenCV**
- SliTraNet: `opencv-contrib-python-headless` (GUI機能なし)
- image_dupp: `opencv-python` (GUI機能あり)
- **解決策**: `opencv-contrib-python` を使用（両方の機能を包含）

**✅ 共通依存関係**:
- `numpy`: 両方で使用、バージョン競合なし

**➕ 新規追加が必要**:
- `imagededup`: dHashアルゴリズムによる画像類似度検出
- `Pillow`: 画像処理ライブラリ
- `tqdm`: プログレスバー表示

#### 統合requirements.txt
```
# PyTorch with CUDA 12.4 support
torch --index-url https://download.pytorch.org/whl/cu124
torchvision --index-url https://download.pytorch.org/whl/cu124

# OpenCV (contrib版で両ツールの要件を満たす)
opencv-contrib-python

# 共通ライブラリ
numpy

# SliTraNet専用
decord

# image_dupp統合用
imagededup
Pillow
tqdm
```

### 統合実装計画

#### Phase 1: image_similarity.py統合
1. **ファイル配置**: `D:\Work\Script\image_dupp\image_similarity.py` を `D:\Work\Script\SliTraNet\` にコピー
2. **依存関係調整**: OpenCV import文の調整
3. **パス処理改善**: SliTraNetの日本語パス対応機能と統合

#### Phase 2: main.py処理フロー拡張
```python
def main():
    # Stage 1: SliTraNet処理（既存）
    video_path = sys.argv[1]
    run_slide_detection(video_path)
    extract_frames(video_path)
    
    # Stage 2: image_dupp処理（新規）
    extracted_frames_dir = os.path.join(os.path.dirname(video_path), "extracted_frames")
    if os.path.exists(extracted_frames_dir):
        run_duplicate_removal(extracted_frames_dir)
        create_final_slides_folder(extracted_frames_dir)
```

#### Phase 3: 出力構造の最適化
```
動画フォルダ/
├── video.mp4
├── video_results.txt           # SliTraNet検出結果
├── extracted_frames/           # SliTraNet抽出結果
│   ├── slide_001_frame_xxx.png
│   ├── slide_002_frame_xxx.png
│   └── dupp/                   # image_dupp移動先
│       ├── slide_003_frame_xxx.png  # 重複画像
│       └── slide_004_frame_xxx.png
├── similarity_groups.txt       # image_dupp解析結果
├── final_slides/               # 最終結果（重複除去後）
│   ├── slide_001_frame_xxx.png
│   ├── slide_002_frame_xxx.png
│   └── slide_005_frame_xxx.png
└── inference.log              # 統合処理ログ
```

### 技術的課題と解決方法

#### 1. dHashアルゴリズムの最適化
**課題**: プレゼンテーション画像特有の類似性検出
**解決**: 
- デフォルト閾値85%を維持
- 連続するスライド間での高精度検出
- SliTraNetのフレーム命名規則に対応

#### 2. ファイル命名の一貫性
**現在のSliTraNet**: `slide_001_frame_000175_00h05m51.234s.png`
**image_dupp対応**: PNG拡張子の自動認識とソート処理

#### 3. エラーハンドリングの統合
- SliTraNetの既存ログ機能を拡張
- image_dupp処理状況をinference.logに統合
- 日本語パス対応の維持

#### 4. パフォーマンス最適化
- dHash計算の並列化（可能であれば）
- 大量画像処理時のメモリ管理
- プログレスバー統合表示

### 実装スケジュール

#### Step 1: 環境準備 🔧 ✅
- [x] 統合requirements.txt作成
- [x] setup.bat更新（新規依存関係インストール）
- [ ] 既存環境での競合テスト

#### Step 2: image_similarity.py統合 📁 ✅
- [x] ファイルコピーと配置
- [x] import文の調整（opencv-contrib-python対応）
- [x] ImageSimilarityAnalyzerクラスの適応

#### Step 3: main.py拡張 🔄 ✅
- [x] duplicate_removal関数追加
- [x] 処理フロー統合（3ステップ化）
- [x] エラーハンドリング拡張

#### Step 4: 出力管理改善 📊 ✅
- [x] extracted_frames/dupp/フォルダ自動生成（final_slides削除）
- [x] similarity_groups.txt出力（動画フォルダ直下）
- [x] ログ統合（inference.log拡張）

#### Step 5: run.bat統合 ⚡ ✅
- [x] 一連の処理を自動実行（3ステップ統合）
- [x] ドラッグ&ドロップ対応維持
- [x] エラー時の適切な停止処理

#### Step 6: テスト・検証 ✅ (実装完了・テスト待ち)
- [ ] 既存動画での動作確認
- [ ] 重複検出精度テスト
- [ ] 処理時間測定
- [ ] 日本語パス対応確認

### 期待される効果

#### 1. 画像数の大幅削減 📉
- SliTraNet: 講演動画60分 → 約50-100枚のスライド画像
- image_dupp: 類似画像除去により → 約20-40枚の最終画像
- **削減率**: 約50-70%の画像数削減

#### 2. ワークフロー改善 🚀
- **Before**: 動画 → SliTraNet → 手動重複除去
- **After**: 動画 → run.bat → 最終結果（自動処理）

#### 3. 品質向上 ✨
- dHashアルゴリズムによる高精度重複検出
- プレゼンテーション特化の最適化
- 一貫したファイル管理

### 実装完了サマリー (2025-07-24) ✅

#### 統合成果
**SliTraNetとimage_duppツールの統合が完了しました！**

**主要な実装変更:**
1. **統合requirements.txt**: OpenCV競合解決、新規依存関係追加
2. **image_similarity.py**: SliTraNet統合版として追加
3. **main.py**: 3ステップ処理フロー（検出→抽出→重複除去）
4. **setup.bat**: 統合依存関係のインストール対応

**統合後の処理フロー:**
```
動画ドロップ → run.bat実行 → 以下の自動処理
├── Step 1: スライド遷移検出
├── Step 2: フレーム抽出 
└── Step 3: 重複画像除去
```

**最終出力構造:**
```
動画フォルダ/
├── video.mp4
├── video_results.txt           # SliTraNet検出結果
├── extracted_frames/           # 最終画像（重複除去後）
│   ├── slide_001_frame_xxx.png
│   ├── slide_002_frame_xxx.png
│   └── dupp/                   # 重複画像移動先
│       ├── slide_003_frame_xxx.png
│       └── slide_004_frame_xxx.png  
├── similarity_groups.txt       # 重複検出結果
└── inference.log              # 統合処理ログ
```

**技術的改良点:**
- ✅ OpenCV競合解決 (`opencv-contrib-python`統一)
- ✅ dHashアルゴリズム統合 (85%閾値)
- ✅ 日本語パス対応維持
- ✅ エラーハンドリング強化
- ✅ プログレス表示統合

**次のステップ**: 実際の動画でのテスト・検証

---

## フレーム抽出処理の高速化改善案 (2025-07-24) 🚀

### 現在の課題
- `extract_frames_in_batches()` でffmpegを使ったバッチ処理
- `batch_size=10` で処理しているが書き出しが遅い
- ffmpegプロセス起動コストとI/O負荷

### 提案: decordによる直接フレーム抽出

#### 技術的メリット
1. **プロセス起動コスト削減**: ffmpegプロセス不要
2. **メモリ効率**: 必要フレームのみをメモリ上で処理
3. **GPU活用**: decordはGPUデコードに対応
4. **バッチ処理**: `vr.get_batch([frame1, frame2, ...])` で一括取得
5. **順序保持**: フレーム番号の順序が自動的に保持される

#### 実装方針

**変更対象**: `extract_frames.py` の `extract_frames_in_batches()` 関数

**新しい処理フロー**:
```python
from decord import VideoReader, cpu, gpu
import cv2
import numpy as np
import os

def extract_frames_with_decord(video_path, frame_indices, output_dir, fps):
    # GPU利用可能時はGPU、そうでなければCPU
    ctx = gpu(0) if torch.cuda.is_available() else cpu(0)
    vr = VideoReader(video_path, ctx=ctx)
    
    # バッチで一括取得（順序が保持される）
    frames = vr.get_batch(frame_indices).asnumpy()
    
    extracted_files = []
    for i, frame in enumerate(frames):
        frame_index = frame_indices[i]
        
        # タイムスタンプ計算
        timestamp = frame_to_timestamp(frame_index, fps)
        
        # ファイル名作成（既存形式互換）
        filename = f"slide_{i+1:03d}_frame_{frame_index:06d}_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        
        # RGB→BGR変換してOpenCV形式で保存
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filepath, frame_bgr)
        
        extracted_files.append(filepath)
    
    return extracted_files
```

#### 期待される性能向上

**処理速度比較 (推定)**:
- **現在**: ffmpegプロセス起動 × 10回 (100フレーム時)
- **改善後**: decord一括処理 × 1回
- **速度向上**: 3-5倍の高速化期待

**メモリ使用量**:
- **現在**: ffmpegプロセス × 10 + 一時ファイル
- **改善後**: VideoReaderインスタンス × 1 + フレームデータ
- **効率化**: メモリ使用量削減

#### 実装上の注意点

1. **GPU対応**: `torch.cuda.is_available()` でGPU/CPU自動切り替え
2. **色空間変換**: decord (RGB) → OpenCV (BGR) 変換必須
3. **エラーハンドリング**: 大量フレーム取得時のメモリ不足対策
4. **互換性維持**: 既存のファイル名形式・タイムスタンプ形式を保持

#### 段階的実装計画

1. **Phase 1**: 既存ffmpeg関数と並行して新関数実装
2. **Phase 2**: 小規模テストでパフォーマンス確認
3. **Phase 3**: 問題なければメイン処理に統合
4. **Phase 4**: 旧ffmpeg関数を削除

#### コード変更箇所

**主要ファイル**: `extract_frames.py`
- `extract_frames_in_batches()` → `extract_frames_with_decord()`
- `get_video_info()` → decord対応版に更新
- `frame_to_timestamp()` → 既存関数流用

**依存関係**: 既存のdecordを活用（追加インストール不要）

#### 実装準備
- 現在のrequirements.txtにdecordが含まれていることを確認済み
- GPUデコード対応でさらなる高速化も期待
- 既存のSliTraNet処理との互換性を維持

---

**実装指示**: この改善案を基に `extract_frames.py` の高速化を実装してください。

---

## 現在の構成 (2025-07-24)

### ファイル構成
- **main.py**: 統合エントリポイント（引数解析、フロー制御）
- **inference_core.py**: スライド遷移検出のコア処理
- **frame_extractor.py**: フレーム抽出処理
- **utils.py**: 共通ユーティリティ関数
- **run.bat**: 統合バッチファイル（動画→推論→フレーム抽出）

### 処理フロー
```
動画入力 → スライド遷移検出 → フレーム抽出 → 結果出力
```

### 使用方法
```bash
# 基本使用（ドラッグ&ドロップ）
run.bat

# コマンドライン
python main.py "video.mp4"

# ROI指定
python main.py "video.mp4" --roi-left-top 0.2 0.1 --roi-right-bottom 0.9 0.8
```

### 出力構造
```
動画フォルダ/
├── video.mp4
├── video_results.txt        # スライド遷移検出結果
├── extracted_frames/        # 抽出フレーム画像
├── inference.log           # 実行ログ
└── video_debug/            # ROI可視化画像（--debugオプション時）
```

## 改良履歴

### Stage 1閾値調整 (2025-07-23)
- `slide_thresh`: 8 → 5（より多くの候補を検出）
- `video_thresh`: 13 → 10（より多くの候補を検出）

### フレーム抽出最適化
- **バッチ処理**: 10個ずつまとめて処理で高速化
- **フィルタリング**: 30フレーム未満のスライドを除外
- **中間フレーム抽出**: 各スライドの代表フレーム1枚のみ

### ROI可視化デバッグ機能
- `--debug`オプションでROI処理結果を可視化
- 動画の5箇所からサンプルフレームを抽出
- 日本語パス対応の画像保存

### 発見された問題点
1. **Stage 2&3のフィルタリング条件が厳しすぎる**
   - 元の条件: `all(slide_transition_pred==3) and all(slide_video_pred==2)`
   - 全ての予測が動画区間でないと除外されないため、実際には除外される候補が少ない

2. **Stage 1の閾値設定が厳しい**
   - `slide_thresh = 8`: 静的スライドの最小長
   - `video_thresh = 13`: 動画区間の最小長
   - これらが動画特性に合わず、初期候補が少なすぎる可能性

3. **長い遷移区間での過剰なクリップ生成**
   - 長い遷移に対して無制限にクリップを生成
   - 予測の一貫性を下げる要因

## 実装した改良点

### 1. Stage 2&3フィルタリング条件の改善 ✅
**変更場所**: `inference.py:314-333`

**改良前**:
```python
if all(slide_transition_pred==3) and all(slide_video_pred==2):
    neg_indices.append(key)
```

**改良後**:
```python
# 多数決ベースのフィルタリング
video_confidence = np.mean(slide_video_pred == 2)
transition_confidence = np.mean(slide_transition_pred == 3)
if video_confidence > 0.6 and transition_confidence > 0.6:
    neg_indices.append(key)
```

**効果**: より現実的な判定基準により、適切な候補の除外が可能

### 2. Stage 1閾値パラメータの調整 ✅
**変更場所**: `inference.py:55-56`

**改良前**:
```python
self.slide_thresh = 8
self.video_thresh = 13
```

**改良後**:
```python
self.slide_thresh = 5  # より多くの候補を検出
self.video_thresh = 10  # より多くの候補を検出
```

**効果**: 初期段階でより多くの候補を検出し、後段での精密な判定に委ねる

### 3. クリップ生成数の制限 ✅
**変更場所**: `data/test_video_clip_dataset.py:85-87`

**改良前**:
```python
k = int(diff/(half_clip_length))+1
```

**改良後**:
```python
max_clips = 5
k = min(int(diff/(half_clip_length))+1, max_clips)
```

**効果**: 長い遷移区間での過剰なクリップ生成を防止し、予測の安定性を向上

## 期待される効果
1. **スライド分割結果の増加**: より緩い初期判定により候補数増加
2. **2回目・3回目推論での過度な減少を抑制**: 改良されたフィルタリング条件
3. **処理の安定性向上**: クリップ数制限による予測ノイズの削減
4. **ログの詳細化**: 信頼度情報の追加により判定過程が可視化

## 追加実装: ROI可視化デバッグ機能 ✅
**変更場所**: `inference.py:96-168, 267`

**目的**: 実際に処理されるROI領域を視覚的に確認するため

**機能**:
- 動画の5箇所（開始、1/4、中央、3/4、終了）からサンプルフレームを抽出
- 各フレームで以下3種類の画像を生成:
  1. `frame_XX_original_with_roi.png`: 元画像にROI境界（緑枠）を描画
  2. `frame_XX_roi_cropped_original.png`: 元解像度でROI部分を切り出し
  3. `frame_XX_roi_cropped_resized.png`: 処理解像度（256×144）でROI部分を切り出し

**技術的課題と解決**:
- **問題**: OpenCVの`cv2.imwrite()`が日本語パスを処理できない
- **解決**: `cv2.imencode()` + `numpy.tofile()`による日本語パス対応保存

**使用方法**:
```python
# 有効化
debug_visualize_roi(video_path, roi, scaled_roi, load_size_roi, pred_dir)

# 無効化（コメントアウト）
"""
debug_visualize_roi(video_path, roi, scaled_roi, load_size_roi, pred_dir)
"""
```

**出力先**: `{動画名}_debug/` (動画と同じフォルダに直接作成)

## 出力構造の変更 ✅ (2025-07-23 追加)
**変更場所**: `inference.py:236-241, 394-437` / `run_inference.bat:79-81`

**変更前の出力構造**:
```
動画フォルダ/
  video.mp4
  video_results/           ← フォルダ作成
    video_results.txt      ← Stage1詳細結果
    video_transitions.txt  ← 検出された遷移
```

**変更後の出力構造**:
```
動画フォルダ/
  video.mp4
  video_results.txt        ← Stage1詳細結果（直接出力）
  inference.log           ← 実行ログ
  video_debug/            ← デバッグ画像用（必要時のみ）
```

**変更理由と効果**:
- **ユーザビリティ向上**: 結果ファイルが動画と同じ場所で即座に確認可能
- **管理の簡素化**: 不要なフォルダ階層を除去
- **transitions.txt廃止**: 冗長な出力ファイルを削減、必要な情報はログで確認可能

## フレーム抽出ツール高速バッチ処理実装 ✅
**変更場所**: `extract_frames.py:全面書き換え` / `extract_frames.bat:表示メッセージ更新`

**目的**: 長いスライド区間（30+フレーム）の中間フレーム1枚を高速バッチ抽出

**主要改良内容**:

### 1. フィルタリング条件の強化
- **動画区間除外**: Slide No = -1 を自動スキップ
- **短期間スライド除外**: フレーム差30未満を自動スキップ
- **中間フレーム抽出**: 各スライドの代表フレーム1枚のみ

### 2. 抽出方式の最適化
- **中間フレーム計算**: `middle_frame = (frame_id0 + frame_id1) / 2`
- **1スライド1フレーム**: 適切なファイル数で管理しやすい
- **タイムスタンプ付きファイル名**: 時刻情報で分析可能

### 3. 高速バッチ処理実装 ⚡
**10個上限バッチ処理**:
```python
def extract_frames_in_batches(video_file, middle_frames, output_dir, fps, batch_size=10):
    # 10個ずつまとめて処理で大幅高速化
    select_parts = [f"eq(n\\,{info['middle_frame']})" for info in batch]
    select_expr = '+'.join(select_parts)  # "eq(n\,10)+eq(n\,20)+eq(n\,30)"
```

**最適化ffmpegコマンド**:
```bash
ffmpeg -i input.mp4 -vf "select='eq(n\,10)+eq(n\,20)+eq(n\,30)'" -vsync 0 frame_%03d.png
```

### 4. 技術的改良
**動画情報取得**:
```python
def get_video_info(video_file):
    # ffprobeでFPS、総フレーム数、時間を取得
    return {'fps': 29.97, 'total_frames': 100800, 'duration': 3363.4}
```

**バッチ処理フロー**:
1. 10個のフレーム番号を組み合わせてselect式構築
2. 1回のffmpeg実行で10フレーム抽出
3. 一時ファイルをタイムスタンプ付きでリネーム
4. 次のバッチを処理

### 5. 出力改良
- **固定フォルダ**: `extracted_frames/` (タイムスタンプなし)
- **詳細進捗**: バッチごとの処理状況表示
- **ファイル名**: `slide_001_frame_000175_00h05m51.234s.png`

### 6. 性能向上効果
- **ffmpegプロセス数**: 1/10に削減 (100フレーム → 10回実行)
- **メモリ効率**: 大量処理でも安定
- **エラー分離**: バッチ単位で失敗を分離
- **速度低下防止**: 段々遅くなる問題を解決

**処理例**:
```
# _results.txtファイルの処理
Slide No, FrameID0, FrameID1
-1, 1, 112      # 動画区間 → スキップ
0, 114, 119     # 差5フレーム → スキップ（30未満）
1, 150, 200     # 差50フレーム → 中間フレーム175を抽出 ✅
2, 350, 450     # 差100フレーム → 中間フレーム400を抽出 ✅

バッチ処理:
Batch 1: frames 175, 400, ... (最大10個)
ffmpeg -i video.mp4 -vf "select='eq(n\,175)+eq(n\,400)+...'" -vsync 0 temp_%03d.png
```

## テスト推奨事項
- 従来問題があった動画での分割結果比較
- Stage別の候補数推移の確認
- 信頼度スコアの分布確認
- ROI設定の妥当性確認（デバッグ画像による視覚的検証）
- **新機能テスト**:
  - `_results.txt`での長期スライド中間フレーム抽出
  - タイムスタンプの正確性確認
  - バッチ処理の性能測定（速度低下防止）
  - 30フレーム未満スライドの除外確認
  - 10個バッチでの安定性テスト

---

# 動画フレーム抽出ツール実装計画（旧計画）

## 概要
transitions.txtファイルの内容を解析し、各トランジション間の中間フレームを動画から抽出してPNG画像として保存するツールを作成する。

## 要件
- ドラッグ&ドロップで2つのファイルを受け取る
  - transitions.txt
  - 対応する動画ファイル（.mp4）
- transitions.txtの各行間の中間フレームを抽出
- PNG形式で出力
- ファイル名にタイムスタンプを付与
- 出力先：transitions.txtと同じフォルダ

## transitions.txtフォーマット
```
Transition No, FrameID0, FrameID1
1, 245, 247
2, 488, 489
...
```

## 実装アプローチ

### 1. Python スクリプト (extract_frames.py)
- transitions.txtを読み込み、パース
- 各行間の中間フレーム番号を計算
- ffmpegを使用して指定フレームから画像を抽出
- タイムスタンプ付きファイル名で保存

### 2. Batファイル (extract_frames.bat)
- ドラッグ&ドロップされた2つのファイルを受け取り
- transitions.txtと動画ファイルを特定
- Pythonスクリプトに引数として渡す

## 処理フロー
1. transitions.txtを読み込み、ヘッダーをスキップ
2. 各行のFrameID1を取得
3. 次の行のFrameID0との中間フレーム番号を計算
4. ffmpegでフレーム抽出
5. タイムスタンプ付きで保存

## 例
- Line1: 1, 245, 247
- Line2: 2, 488, 489
- 中間フレーム: (247 + 488) / 2 = 367.5 → 368フレーム目を抽出

## 技術的詳細
- ffmpegの-vfオプションでselect='eq(n,フレーム番号)'を使用
- 出力ファイル名：`frame_{フレーム番号}_{タイムスタンプ}.png`
- エラーハンドリング：ファイル存在チェック、ffmpeg実行結果確認