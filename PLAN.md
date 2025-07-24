# SliTraNet 統合実装記録

## 統合実装 (2025-07-24)

### 統合の背景と目標
個別実行していた`run_inference.bat`と`extract_frames.bat`を統合し、動画入力から画像分類まで一貫した処理を提供する。

### 実装内容

#### 1. コード分割・リファクタリング ✅
**目的**: 400行制限を遵守しつつ、機能分離と保守性向上

**分割構成**:
- **main.py** (296行): 統合エントリポイント、引数解析、エラーハンドリング
- **inference_core.py** (256行): スライド遷移検出のコア処理
- **frame_extractor.py** (200行): フレーム抽出処理
- **utils.py** (154行): 共通ユーティリティ関数

**設計原則**:
- 単一責任の原則: 各モジュールが明確な役割を持つ
- 依存関係の整理: inference_core → frame_extractor → utils
- エラーハンドリングの統一: ログ出力とエラー伝播を一元管理

#### 2. 統合エントリポイント (main.py) ✅
**機能**:
```
動画入力 → スライド遷移検出 → フレーム抽出 → 結果出力
```

**主要処理フロー**:
1. **引数検証**: 動画ファイルとROI座標の妥当性確認
2. **Step 1**: `run_slide_detection()` - スライド遷移検出
3. **Step 2**: `extract_slide_frames()` - フレーム抽出
4. **結果統合**: 成功/失敗の統一的な報告

**外部引数対応**:
```bash
python main.py video.mp4 --roi-left-top 0.2 0.1 --roi-right-bottom 0.9 0.8 --debug
```

#### 3. 統合バッチファイル (run.bat) ✅
**統合前**:
- `run_inference.bat` → `_results.txt`
- `extract_frames.bat` → `extracted_frames/`

**統合後**:
- `run.bat` → 動画から画像まで自動処理

**エラーハンドリング**:
- 推論失敗時はフレーム抽出をスキップして停止
- 詳細なエラー情報をログファイルに記録

#### 4. 設計改良点

**4.1 依存関係チェック**:
```python
def check_dependencies():
    # Pythonモジュール確認
    # 外部コマンド(ffmpeg)確認
    return missing_deps
```

**4.2 統一ログシステム**:
```python
def setup_logging():
    # ファイル + コンソール出力
    # UTF-8エンコーディング対応
```

**4.3 中間ファイル管理**:
- `_results.txt`は保持（ユーザー要求）
- 統合処理でのファイル依存関係を明確化

### 統合後の使用方法

#### 基本使用（ドラッグ&ドロップ）:
```
動画ファイル → run.bat へドラッグ&ドロップ
```

#### コマンドライン使用:
```bash
run.bat "video.mp4"
# または直接
python main.py "video.mp4"
```

#### 高度な使用（ROI指定）:
```bash
python main.py "video.mp4" --roi-left-top 0.2 0.1 --roi-right-bottom 0.9 0.8
```

### 統合後の出力構造
```
動画フォルダ/
├── video.mp4
├── video_results.txt        # スライド遷移検出結果
├── extracted_frames/        # 抽出フレーム画像
│   ├── slide_001_frame_000175_00h05m51.234s.png
│   └── slide_002_frame_000400_00h13m20.123s.png
├── inference.log           # 統合処理ログ
└── video_debug/            # ROI可視化画像（--debugオプション時）
```

### 次段階への対応
- **外部関数化**: 引数による設定変更への対応完了
- **モジュラー設計**: 機能追加・変更が容易な構造
- **400行制限**: 全モジュールが制限内で計画的分割済み

## 以前の改良記録 (2025-07-23)

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