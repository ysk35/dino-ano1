# AnomalyDINO 2段階異常検知実装計画書

## 📋 概要

本文書は、AnomalyDINOの精度向上のための2段階異常検知手法の実装計画を記載する。
既存のパッチベース局所異常検知に統計量ベースの全体異常検知を追加し、構造的・配置的異常の検知能力向上を目指す。

---

## 🎯 現状分析

### ベースライン性能
- **全体**: 多くのクラスでAUROC>0.98, F1>0.95の高性能
- **完全正解**: carpet, leather (F1=1.00)
- **高性能**: bottle, grid, metal_nut, tile, toothbrush, wood, zipper (F1≳0.96)

### 特定された問題点

#### 1. 構造的・配置的異常の検知困難
```
cable_swap: 12/12 FN (100%見逃し)
missing_wire: 4/10 FN (40%見逃し)
```
**原因**: パッチレベル特徴では「正しい配置」を学習できない

#### 2. 視点・角度依存異常の検知困難
```
screw: manipulated_front (13 FN), thread_side (7 FN)
```
**原因**: DINOv2特徴の視点不変性が裏目

#### 3. 微細局所異常の限界
```
capsule: crack (5 FN), faulty_imprint (5 FN)
```

#### 4. 正常内バリエーション過敏
```
pill: good 7/26 FP
transistor: good 8/60 FP
```

---

## 💡 提案手法：2段階異常検知

### 基本アイデア
現在の「パッチレベル局所異常検知」に「統計量ベース全体異常検知」を追加し、補完的な異常検知を実現。

### アーキテクチャ概要
```
Stage 1: 既存のTop1%スコア (パッチベース局所異常検知)
         ↓ (threshold_1を超えない場合)
Stage 2: 統計量スコア (全体統計による補完検知)
         ↓
最終判定: OR論理による統合
```

### 期待される改善
- **cable_swap**: 配置パターンの統計的変化で検知
- **missing_wire**: 部品欠損による全体統計変化で検知  
- **manipulated_front**: 形状分布の変化で検知

---

## 🔧 実装内容

### 1. 統計量メモリバンク構築

**場所**: `src/detection.py` の `run_anomaly_detection()` 関数

```python
# 既存のメモリバンク構築ループ内に追加
stats_ref_mean = []  # 平均ベクトル集合
stats_ref_std = []   # 標準偏差ベクトル集合

for img_ref_n in tqdm(img_ref_samples, desc="Building memory bank", leave=False):
    # ... 既存のパッチ特徴抽出 ...
    
    masked_features = features_ref_i[mask_ref]
    
    # 統計量計算
    patch_mean = masked_features.mean(axis=0)  # [1024] 全体平均特徴
    patch_std = masked_features.std(axis=0)    # [1024] 全体ばらつき特徴
    
    # L2正規化
    patch_mean = patch_mean / (np.linalg.norm(patch_mean) + 1e-8)
    patch_std = patch_std / (np.linalg.norm(patch_std) + 1e-8)
    
    stats_ref_mean.append(patch_mean)
    stats_ref_std.append(patch_std)

# 配列変換
stats_ref_mean = np.array(stats_ref_mean).astype('float32')
stats_ref_std = np.array(stats_ref_std).astype('float32')
```

### 2. 統計量インデックス構築

```python
# 既存のFaissインデックス構築後に追加
if faiss_on_cpu:
    stats_index_mean = faiss.IndexFlatL2(stats_ref_mean.shape[1])
    stats_index_std = faiss.IndexFlatL2(stats_ref_std.shape[1])
else:
    res = faiss.StandardGpuResources()
    stats_index_mean = faiss.GpuIndexFlatL2(res, stats_ref_mean.shape[1])
    stats_index_std = faiss.GpuIndexFlatL2(res, stats_ref_std.shape[1])

# 正規化・追加
faiss.normalize_L2(stats_ref_mean)
faiss.normalize_L2(stats_ref_std)
stats_index_mean.add(stats_ref_mean)
stats_index_std.add(stats_ref_std)
```

### 3. テスト時統計量スコア計算

```python
# 既存の距離計算後に追加

# テスト画像の統計量計算
test_mean = features2.mean(axis=0)
test_std = features2.std(axis=0)

# L2正規化
test_mean = test_mean / (np.linalg.norm(test_mean) + 1e-8)
test_std = test_std / (np.linalg.norm(test_std) + 1e-8)

# 統計量距離計算
faiss.normalize_L2(test_mean.reshape(1, -1))
faiss.normalize_L2(test_std.reshape(1, -1))

dist_mean, _ = stats_index_mean.search(test_mean.reshape(1, -1), k=1)
dist_std, _ = stats_index_std.search(test_std.reshape(1, -1), k=1)

# 統計量スコア計算（コサイン距離変換）
stats_score = 0.7 * (dist_mean[0][0] / 2) + 0.3 * (dist_std[0][0] / 2)
```

### 4. 2段階判定ロジック

```python
def two_stage_detection(patch_score, stats_score, object_name):
    """2段階異常検知"""
    
    # カテゴリ別閾値設定
    thresholds = {
        'cable': {'t1': 0.15, 't2': 0.08},      # 積極的Stage2
        'screw': {'t1': 0.18, 't2': 0.12},      # 標準
        'capsule': {'t1': 0.16, 't2': 0.10},    # 標準
        'pill': {'t1': 0.14, 't2': 0.20},       # 保守的Stage2
        'transistor': {'t1': 0.17, 't2': 0.18}, # 保守的Stage2
        'default': {'t1': 0.16, 't2': 0.12}
    }
    
    thres = thresholds.get(object_name, thresholds['default'])
    
    # Stage 1: パッチベース検知
    if patch_score > thres['t1']:
        return True, patch_score, "patch_based"
    
    # Stage 2: 統計量による補完検知
    elif stats_score > thres['t2']:
        return True, stats_score, "statistics_based"
    
    # 正常判定
    else:
        return False, max(patch_score, stats_score), "normal"

# メインロジックでの使用
patch_score = mean_top1p(output_distances.flatten())
is_anomaly, final_score, detection_method = two_stage_detection(
    patch_score, stats_score, object_name
)

anomaly_scores[f"{type_anomaly}/{img_test_nr}"] = final_score
```

---

## 📊 期待される効果

### 定量的改善予測

#### Best Case シナリオ
```
cable: F1 0.875 → 0.95 (+cable_swap完全解決)
screw: F1 0.895 → 0.92 (+orientation異常の50%改善)
全体: AUROC +0.5-0.8%改善
```

#### Realistic シナリオ  
```
cable: F1 0.875 → 0.90 (+cable_swapの60%改善)
screw: F1 0.895 → 0.91 (+orientation異常の30%改善)
全体: AUROC +0.2-0.4%改善
```

### カテゴリ別期待効果
- **cable**: Stage2で構造的異常を大幅改善
- **screw**: 視点変化異常を部分改善
- **capsule**: 印字異常を微改善
- **pill/transistor**: FP抑制のため保守的設定

---

## 🚀 実装スケジュール

### Phase 1: 基本実装 (1-2日)
- [ ] 統計量メモリバンク構築
- [ ] 統計量インデックス構築  
- [ ] 2段階検知ロジック実装
- [ ] 基本動作確認

### Phase 2: カテゴリ別最適化 (2-3日)
- [ ] カテゴリ別閾値調整
- [ ] cable_swapでの詳細検証
- [ ] FP/FN バランス調整

### Phase 3: 詳細評価・分析 (3-4日)
- [ ] 全15カテゴリでの性能評価
- [ ] 異常タイプ別改善分析
- [ ] Stage別貢献度分析

### Phase 4: 論文向けまとめ (1-2日)
- [ ] 結果の統計的検定
- [ ] 可視化・図表作成
- [ ] 考察・限界分析

---

## 🧪 評価方法

### 主要評価指標
1. **全体性能**: AUROC, F1, Precision, Recall
2. **Stage別分析**: Stage1のみ/Stage2救済/Stage2 FP
3. **異常タイプ別**: cable_swap, missing_wire等の個別改善

### 成功判定基準

#### 必達目標
- cable_swap: 12 FN → 6 FN以下 (50%以上改善)
- 全体AUROC: +0.2%以上改善
- FP増加: 現状の20%以下

#### 理想目標
- cable_swap: 12 FN → 3 FN以下 (75%以上改善)
- 全体AUROC: +0.4%以上改善
- screw, capsuleでも明確な改善

---

## 🔍 実装時の注意点

### 1. 計算効率
- 統計量計算は軽量（平均・標準偏差のみ）
- メモリ使用量増加は最小限（参照画像数 × 2 × 1024次元）
- 推論時間増加は<5%

### 2. 閾値設定
- 正常画像での統計量分布から適応的に設定
- カテゴリ特性に応じた重み調整
- FP増加を抑制する保守的設定

### 3. デバッグ・検証
- Stage別検知数の記録
- 統計量スコア分布の確認
- 特定異常タイプでの詳細分析

---

## 📚 参考文献・理論的背景

1. **パッチベース手法の限界**: 局所特徴による構造的異常検知の困難性
2. **統計的異常検知**: 分布の変化による異常検知の有効性
3. **補完的アプローチ**: 異なる特徴量の組み合わせによる性能向上

---

## 📝 修士論文での位置づけ

### 学術的貢献
1. **既存手法の限界分析**: パッチベース手法が苦手な異常タイプの特定
2. **理論的補完手法**: 局所vs全体特徴の相補性の実証
3. **実用的改善**: 軽量で実装容易な改善手法の提案

### 期待される成果
- 明確な問題設定と解決策の提示
- 特定異常タイプでの劇的改善（cable_swap等）
- 工学的価値の高い実用的手法

---

**実装者へのメッセージ**: 
この実装は理論的根拠が明確で、特にcable_swapでの劇的改善が期待できます。
段階的に実装し、各フェーズで効果を確認しながら進めることで、
確実に成果を得られる計画となっています。