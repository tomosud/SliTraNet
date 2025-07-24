# SliTraNet 統合実装記録

## 統合システム概要 (2025-07-24)

SliTraNetとimage_duppツールを統合し、動画からスライドフレーム抽出→重複除去まで一括処理するシステム。

### 処理フロー
```
動画入力 → スライド遷移検出 → フレーム抽出 → 重複除去 → 最終結果
    ↓              ↓              ↓          ↓          ↓
 run.bat    inference_core.py  frame_extractor.py  image_similarity.py  extracted_frames/
```

### 主要ファイル構成
```
SliTraNet/
├── main.py                    # 統合エントリポイント
├── inference_core.py          # スライド遷移検出
├── frame_extractor.py         # decordフレーム抽出（日本語パス対応）
├── image_similarity.py        # 重複画像検出・除去
├── utils.py                   # 共通ユーティリティ
├── run.bat                    # ドラッグ&ドロップ実行
├── test_extract.bat/py        # decord検証ツール
└── requirements.txt           # 統合依存関係
```

## 技術実装詳細

### 1. 依存関係統合 ✅
**OpenCV競合解決**: `opencv-contrib-python`で統一（GUI機能 + contrib機能）

**統合requirements.txt**:
```
torch --index-url https://download.pytorch.org/whl/cu124
torchvision --index-url https://download.pytorch.org/whl/cu124
opencv-contrib-python
numpy
decord
imagededup
Pillow
tqdm
```

### 2. decord高速フレーム抽出 ✅
**主な改善**:
- ffmpegプロセス起動コスト削減（10回→1回）
- GPU/CPU自動切り替え（CUDA失敗時CPU fallback）
- 日本語パス対応保存（`cv2.imencode() + numpy.tofile()`）
- バッチ一括取得で大幅高速化

**実装場所**: `frame_extractor.py:extract_frames_with_decord()`

```python
def extract_frames_with_decord(video_file, middle_frames, output_dir, fps):
    # GPU試行→CPU fallback
    if torch.cuda.is_available():
        try:
            ctx = gpu(0)
            vr = VideoReader(video_path, ctx=ctx)
        except:
            ctx = cpu(0)
            vr = VideoReader(video_path, ctx=ctx)
    
    # 一括フレーム取得
    frames = vr.get_batch(frame_indices).asnumpy()
    
    # 日本語パス対応保存
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        save_image_japanese_path(frame_bgr, filepath)
```

### 3. 日本語パス対応 ✅
**問題**: OpenCVの`cv2.imwrite()`が日本語パスで失敗
**解決**: `cv2.imencode() + numpy.tofile()`による代替保存

```python
def save_image_japanese_path(image, filepath):
    try:
        # 方法1: 直接保存
        success = cv2.imwrite(filepath, image)
        if success: return True
    except:
        pass
    
    # 方法2: エンコード→ファイル書き出し
    success, encoded_img = cv2.imencode('.png', image)
    if success:
        encoded_img.tofile(filepath)
        return True
    return False
```

### 4. 重複画像除去統合 ✅
**実装**: `image_similarity.py`でdHashアルゴリズム（85%閾値）
**処理**: 類似画像を`extracted_frames/dupp/`に移動
**出力**: `similarity_groups.txt`で重複情報記録

### 5. 検証ツール ✅
**test_extract.bat/py**: decord機能の独立検証
- 動画を100等分して100枚抽出
- GPU/CPU切り替えテスト
- 日本語パス保存テスト
- 詳細エラーログ

## 出力構造

```
動画フォルダ/
├── video.mp4
├── video_results.txt           # スライド検出結果
├── extracted_frames/           # 最終画像（重複除去後）
│   ├── slide_001_frame_xxx.png
│   ├── slide_002_frame_xxx.png
│   └── dupp/                   # 重複画像移動先
│       ├── slide_003_frame_xxx.png
│       └── slide_004_frame_xxx.png
├── similarity_groups.txt       # 重複検出詳細
└── inference.log              # 処理ログ
```

## 性能改善効果

### 処理速度
- **ffmpeg→decord**: 3-5倍高速化（プロセス起動コスト削減）
- **バッチ処理**: 10個→1回実行で大幅効率化
- **GPU加速**: 利用可能時はGPUデコード

### 画像数削減
- **SliTraNet**: 動画60分→50-100枚
- **重複除去**: 最終20-40枚（50-70%削減）

### 使いやすさ
- **ワンクリック**: `run.bat`にドラッグ&ドロップ
- **日本語対応**: パス・ファイル名の制限なし
- **自動処理**: 検出→抽出→重複除去まで全自動

## 主要改良履歴

### Stage 1-3 閾値最適化 ✅
- `slide_thresh`: 8→5, `video_thresh`: 13→10
- フィルタリング条件改善（多数決ベース）
- クリップ生成数制限（最大5個）

### フレーム抽出最適化 ✅
- 30フレーム未満スライド除外
- 中間フレーム1枚抽出方式
- タイムスタンプ付きファイル名

### エラーハンドリング強化 ✅
- GPU初期化失敗時CPU自動切り替え
- 日本語パス保存失敗時代替手法
- 詳細ログ出力とステータス表示

## 今後の改善点

### 1. さらなる高速化
- 並列フレーム処理
- メモリ最適化
- GPU利用率向上

### 2. 検出精度向上
- 閾値の動的調整
- 動画特性に応じた最適化
- 機械学習モデルの改良

### 3. 使いやすさ向上
- GUI化の検討
- バッチ処理対応
- 設定ファイル対応