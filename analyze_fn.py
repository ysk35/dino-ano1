#!/usr/bin/env python3
"""
False Negative (見逃し) 分析ツール

詳細スコアファイルを読み込んで、見逃した異常サンプルを分析し、
Stage1/Stage2のどちらで検知可能かを確認する。
"""

import json
import argparse
import numpy as np
from collections import defaultdict


def load_detailed_scores(json_file):
    """詳細スコアファイルを読み込む"""
    with open(json_file, 'r') as f:
        return json.load(f)


def analyze_fn(detailed_scores, stage1_threshold=0.16, stage2_threshold=0.12):
    """False Negativeを分析"""

    fn_samples = []
    tp_samples = []
    fp_samples = []
    tn_samples = []

    # 分類
    for sample_name, scores in detailed_scores.items():
        gt_label = scores['gt_label']
        is_anomaly = scores['is_anomaly']

        if gt_label and not is_anomaly:
            fn_samples.append((sample_name, scores))
        elif gt_label and is_anomaly:
            tp_samples.append((sample_name, scores))
        elif not gt_label and is_anomaly:
            fp_samples.append((sample_name, scores))
        else:
            tn_samples.append((sample_name, scores))

    # 統計情報
    print("=" * 80)
    print("📊 検知結果サマリー")
    print("=" * 80)
    print(f"True Positive (TP):  {len(tp_samples):3d} - 異常を正しく検知")
    print(f"False Negative (FN): {len(fn_samples):3d} - 異常を見逃し ⚠️")
    print(f"False Positive (FP): {len(fp_samples):3d} - 正常を誤検知")
    print(f"True Negative (TN):  {len(tn_samples):3d} - 正常を正しく判定")
    print()

    if len(fn_samples) == 0:
        print("✅ 見逃しサンプルなし！完璧な検知です。")
        return

    # FN詳細分析
    print("=" * 80)
    print(f"🔍 False Negative 詳細分析 ({len(fn_samples)}サンプル)")
    print("=" * 80)
    print()

    # 異常タイプ別にグループ化
    fn_by_type = defaultdict(list)
    for sample_name, scores in fn_samples:
        anomaly_type = sample_name.split('/')[0]
        fn_by_type[anomaly_type].append((sample_name, scores))

    # 異常タイプ別に分析
    for anomaly_type, samples in sorted(fn_by_type.items()):
        print(f"\n📁 {anomaly_type} ({len(samples)}サンプル)")
        print("-" * 80)

        stage1_catchable = 0
        stage2_catchable = 0
        both_miss = 0

        for sample_name, scores in samples:
            patch_score = scores['patch_score']
            stats_score = scores['stats_score']

            # 現在の閾値で検知可能かチェック
            can_catch_stage1 = patch_score > stage1_threshold
            can_catch_stage2 = stats_score > stage2_threshold

            if can_catch_stage1:
                stage1_catchable += 1
                status = "✓ Stage1でキャッチ可能"
            elif can_catch_stage2:
                stage2_catchable += 1
                status = "✓ Stage2でキャッチ可能"
            else:
                both_miss += 1
                status = "✗ 両Stage共に低スコア"

            print(f"  {sample_name:40s} | "
                  f"Stage1: {patch_score:.4f} | "
                  f"Stage2: {stats_score:.4f} | "
                  f"{status}")

        print()
        print(f"  【{anomaly_type} まとめ】")
        print(f"    Stage1で検知可能: {stage1_catchable}/{len(samples)}")
        print(f"    Stage2で検知可能: {stage2_catchable}/{len(samples)}")
        print(f"    両方とも低スコア: {both_miss}/{len(samples)}")

    # 閾値調整シミュレーション
    print("\n" + "=" * 80)
    print("🎯 閾値調整シミュレーション")
    print("=" * 80)

    # Stage1閾値を下げた場合
    thresholds_to_test = [0.10, 0.12, 0.14, 0.16, 0.18, 0.20]

    print("\n【Stage1閾値を変更した場合】")
    print(f"{'閾値':>8s} | {'FN削減':>8s} | {'残りFN':>8s} | {'新FP増加予測':>14s}")
    print("-" * 50)

    for t1 in thresholds_to_test:
        fn_reduced = sum(1 for _, scores in fn_samples if scores['patch_score'] > t1)
        fn_remaining = len(fn_samples) - fn_reduced
        # FP増加の簡易推定（TNのうち閾値を超えるもの）
        new_fp = sum(1 for _, scores in tn_samples if scores['patch_score'] > t1)

        print(f"{t1:>8.2f} | {fn_reduced:>8d} | {fn_remaining:>8d} | {new_fp:>14d}")

    print("\n【Stage2閾値を変更した場合】")
    print(f"{'閾値':>8s} | {'FN削減':>8s} | {'残りFN':>8s} | {'新FP増加予測':>14s}")
    print("-" * 50)

    for t2 in thresholds_to_test:
        # Stage1で検知できなかったFNのみ対象
        fn_stage1_miss = [(name, scores) for name, scores in fn_samples
                          if scores['patch_score'] <= stage1_threshold]
        fn_reduced = sum(1 for _, scores in fn_stage1_miss if scores['stats_score'] > t2)
        fn_remaining = len(fn_samples) - fn_reduced
        # FP増加の簡易推定
        tn_stage1_miss = [(name, scores) for name, scores in tn_samples
                          if scores['patch_score'] <= stage1_threshold]
        new_fp = sum(1 for _, scores in tn_stage1_miss if scores['stats_score'] > t2)

        print(f"{t2:>8.2f} | {fn_reduced:>8d} | {fn_remaining:>8d} | {new_fp:>14d}")

    # スコア分布統計
    print("\n" + "=" * 80)
    print("📈 スコア分布統計")
    print("=" * 80)

    fn_patch_scores = [scores['patch_score'] for _, scores in fn_samples]
    fn_stats_scores = [scores['stats_score'] for _, scores in fn_samples]
    tp_patch_scores = [scores['patch_score'] for _, scores in tp_samples]
    tp_stats_scores = [scores['stats_score'] for _, scores in tp_samples]

    print(f"\n【FN (見逃し) のスコア分布】")
    print(f"  Stage1 (patch): 平均={np.mean(fn_patch_scores):.4f}, "
          f"中央値={np.median(fn_patch_scores):.4f}, "
          f"最大={np.max(fn_patch_scores):.4f}, "
          f"最小={np.min(fn_patch_scores):.4f}")
    print(f"  Stage2 (stats): 平均={np.mean(fn_stats_scores):.4f}, "
          f"中央値={np.median(fn_stats_scores):.4f}, "
          f"最大={np.max(fn_stats_scores):.4f}, "
          f"最小={np.min(fn_stats_scores):.4f}")

    print(f"\n【TP (正常検知) のスコア分布】")
    print(f"  Stage1 (patch): 平均={np.mean(tp_patch_scores):.4f}, "
          f"中央値={np.median(tp_patch_scores):.4f}, "
          f"最大={np.max(tp_patch_scores):.4f}, "
          f"最小={np.min(tp_patch_scores):.4f}")
    print(f"  Stage2 (stats): 平均={np.mean(tp_stats_scores):.4f}, "
          f"中央値={np.median(tp_stats_scores):.4f}, "
          f"最大={np.max(tp_stats_scores):.4f}, "
          f"最小={np.min(tp_stats_scores):.4f}")

    # Stage2の貢献度分析
    print("\n" + "=" * 80)
    print("🎖️  Stage2 貢献度分析")
    print("=" * 80)

    stage2_detections = sum(1 for _, scores in tp_samples
                            if scores['detection_method'] == 'statistics_based')
    stage1_detections = sum(1 for _, scores in tp_samples
                            if scores['detection_method'] == 'patch_based')

    print(f"\n【検知方法の内訳】")
    print(f"  Stage1 (patch) のみで検知: {stage1_detections}/{len(tp_samples)} "
          f"({100*stage1_detections/len(tp_samples):.1f}%)")
    print(f"  Stage2 (stats) で救済検知: {stage2_detections}/{len(tp_samples)} "
          f"({100*stage2_detections/len(tp_samples):.1f}%)")

    if stage2_detections == 0:
        print("\n⚠️  Stage2での検知が0件です！")
        print("   → Stage2閾値が高すぎる可能性があります")
        print("   → または統計量スコアが全体的に低い可能性があります")


def main():
    parser = argparse.ArgumentParser(
        description='False Negative分析ツール - 見逃した異常を詳細分析')
    parser.add_argument('json_file', type=str,
                        help='詳細スコアのJSONファイル (e.g., detailed_scores_cable_5shot.json)')
    parser.add_argument('--stage1_threshold', type=float, default=0.16,
                        help='Stage1閾値 (default: 0.16)')
    parser.add_argument('--stage2_threshold', type=float, default=0.12,
                        help='Stage2閾値 (default: 0.12)')

    args = parser.parse_args()

    print(f"\n📂 ファイル: {args.json_file}")
    print(f"🎯 現在の閾値: Stage1={args.stage1_threshold}, Stage2={args.stage2_threshold}")
    print()

    detailed_scores = load_detailed_scores(args.json_file)
    analyze_fn(detailed_scores, args.stage1_threshold, args.stage2_threshold)

    print("\n" + "=" * 80)
    print("分析完了！")
    print("=" * 80)


if __name__ == "__main__":
    main()
