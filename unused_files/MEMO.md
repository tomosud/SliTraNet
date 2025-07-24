# 未使用ファイル整理メモ

## 移動したファイル

### test_SliTraNet.py
- **理由**: batファイルからの実行フローで使用されていない
- **内容**: 
  - `printLog()` - inference.pyと重複
  - `detect_slide_transitions()` - inference.pyと重複  
  - `test_SliTraNet()` - テスト関数、実際の実行では呼ばれない
- **移動日**: 2025-07-24
- **状態**: 完全に未使用、削除可能

## 削除した関数

### test_slide_detection_2d.py
- `test_resnet2d()` - テスト関数、実際の実行では呼ばれない（削除済み）
- **注意**: `detect_initial_slide_transition_candidates_resnet2d()`は inference.py で使用中のため削除不可

## 使用状況の分析結果

### 実際に使用されるファイル（batファイル経由）
1. **inference.py** (run_inference.bat → python inference.py)
2. **extract_frames.py** (extract_frames.bat → python extract_frames.py)
3. **model.py** (inference.pyから import)
4. **test_slide_detection_2d.py** (inference.pyから部分的に import)

### 実行フロー
```
run_inference.bat
└── inference.py
    ├── model.py (define_resnet2d, loadNetwork)
    └── test_slide_detection_2d.py (detect_initial_slide_transition_candidates_resnet2d)

extract_frames.bat
└── extract_frames.py (独立実行)
```

## 推論処理の最適化履歴

### 2025-07-24: Stage 2・3処理のコメントアウト
**変更理由**: _results.txt出力のみが必要で、Stage 2・3の追加推論が不要なため

**変更内容**:
- **Stage 1**: 初期スライド遷移候補検出 → `_results.txt`出力 (**実行継続**)
- **Stage 2**: スライド-動画判定 (**コメントアウト**)
- **Stage 3**: スライド遷移検出 (**コメントアウト**)

**効果**:
- 処理時間の大幅短縮
- 不要なGPU計算の削減
- _results.txtまでの出力は維持

**復元方法**:
Stage 2・3が必要な場合は、inference.py内の該当コメントアウト部分を有効化

```python
# 現在: Stage 1完了後にreturn True
# 復元: コメントアウト箇所の # を削除
```