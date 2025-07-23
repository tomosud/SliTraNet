# SliTraNet 改良記録

## 問題分析 (2025-07-23)

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

## テスト推奨事項
- 従来問題があった動画での分割結果比較
- Stage別の候補数推移の確認
- 信頼度スコアの分布確認

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