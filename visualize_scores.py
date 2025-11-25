#!/usr/bin/env python3
"""
スコア分布可視化ツール

詳細スコアファイルを読み込んで、Stage1/Stage2のスコア分布を可視化する。
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def load_detailed_scores(json_file):
    """詳細スコアファイルを読み込む"""
    with open(json_file, 'r') as f:
        return json.load(f)


def visualize_scores(detailed_scores, stage1_threshold=0.16, stage2_threshold=0.12, output_file=None):
    """スコア分布を可視化"""

    # データ分類
    fn_samples = []
    tp_samples = []
    fp_samples = []
    tn_samples = []

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

    # スコア抽出
    fn_patch = [scores['patch_score'] for _, scores in fn_samples]
    fn_stats = [scores['stats_score'] for _, scores in fn_samples]
    tp_patch = [scores['patch_score'] for _, scores in tp_samples]
    tp_stats = [scores['stats_score'] for _, scores in tp_samples]
    fp_patch = [scores['patch_score'] for _, scores in fp_samples]
    fp_stats = [scores['stats_score'] for _, scores in fp_samples]
    tn_patch = [scores['patch_score'] for _, scores in tn_samples]
    tn_stats = [scores['stats_score'] for _, scores in tn_samples]

    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Stage1 vs Stage2 スコア分布', fontsize=16, fontweight='bold')

    # 1. 散布図（全サンプル）
    ax = axes[0, 0]
    if len(tp_patch) > 0:
        ax.scatter(tp_patch, tp_stats, c='green', alpha=0.6, s=30, label=f'TP ({len(tp_samples)})')
    if len(fn_patch) > 0:
        ax.scatter(fn_patch, fn_stats, c='red', alpha=0.8, s=50, marker='x', label=f'FN ({len(fn_samples)})')
    if len(fp_patch) > 0:
        ax.scatter(fp_patch, fp_stats, c='orange', alpha=0.5, s=30, label=f'FP ({len(fp_samples)})')
    if len(tn_patch) > 0:
        ax.scatter(tn_patch, tn_stats, c='blue', alpha=0.3, s=20, label=f'TN ({len(tn_samples)})')

    ax.axvline(stage1_threshold, color='purple', linestyle='--', linewidth=2, label=f'Stage1閾値 ({stage1_threshold})')
    ax.axhline(stage2_threshold, color='brown', linestyle='--', linewidth=2, label=f'Stage2閾値 ({stage2_threshold})')
    ax.set_xlabel('Stage1 Score (Patch-based)', fontsize=12)
    ax.set_ylabel('Stage2 Score (Statistics-based)', fontsize=12)
    ax.set_title('全サンプルのスコア分布', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    # 2. Stage1スコアのヒストグラム
    ax = axes[0, 1]
    bins = np.linspace(0, max(tp_patch + fn_patch + [0.3]), 30)
    if len(tp_patch) > 0:
        ax.hist(tp_patch, bins=bins, alpha=0.6, color='green', label=f'TP ({len(tp_samples)})', edgecolor='black')
    if len(fn_patch) > 0:
        ax.hist(fn_patch, bins=bins, alpha=0.8, color='red', label=f'FN ({len(fn_samples)})', edgecolor='black')
    ax.axvline(stage1_threshold, color='purple', linestyle='--', linewidth=2, label=f'閾値 ({stage1_threshold})')
    ax.set_xlabel('Stage1 Score (Patch-based)', fontsize=12)
    ax.set_ylabel('サンプル数', fontsize=12)
    ax.set_title('Stage1 スコア分布', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # 3. Stage2スコアのヒストグラム
    ax = axes[1, 0]
    bins = np.linspace(0, max(tp_stats + fn_stats + [0.3]), 30)
    if len(tp_stats) > 0:
        ax.hist(tp_stats, bins=bins, alpha=0.6, color='green', label=f'TP ({len(tp_samples)})', edgecolor='black')
    if len(fn_stats) > 0:
        ax.hist(fn_stats, bins=bins, alpha=0.8, color='red', label=f'FN ({len(fn_samples)})', edgecolor='black')
    ax.axvline(stage2_threshold, color='brown', linestyle='--', linewidth=2, label=f'閾値 ({stage2_threshold})')
    ax.set_xlabel('Stage2 Score (Statistics-based)', fontsize=12)
    ax.set_ylabel('サンプル数', fontsize=12)
    ax.set_title('Stage2 スコア分布', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # 4. FNのスコア分析（異常タイプ別）
    ax = axes[1, 1]
    if len(fn_samples) > 0:
        fn_by_type = defaultdict(list)
        for sample_name, scores in fn_samples:
            anomaly_type = sample_name.split('/')[0]
            fn_by_type[anomaly_type].append(scores)

        x_labels = []
        stage1_means = []
        stage2_means = []

        for anomaly_type, scores_list in sorted(fn_by_type.items()):
            x_labels.append(f"{anomaly_type}\n({len(scores_list)})")
            stage1_means.append(np.mean([s['patch_score'] for s in scores_list]))
            stage2_means.append(np.mean([s['stats_score'] for s in scores_list]))

        x = np.arange(len(x_labels))
        width = 0.35

        ax.bar(x - width/2, stage1_means, width, label='Stage1平均', color='lightcoral', edgecolor='black')
        ax.bar(x + width/2, stage2_means, width, label='Stage2平均', color='lightblue', edgecolor='black')
        ax.axhline(stage1_threshold, color='purple', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Stage1閾値')
        ax.axhline(stage2_threshold, color='brown', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Stage2閾値')

        ax.set_xlabel('異常タイプ', fontsize=12)
        ax.set_ylabel('平均スコア', fontsize=12)
        ax.set_title(f'FN（見逃し）の異常タイプ別スコア', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'FNなし！\n完璧な検知', ha='center', va='center',
                fontsize=20, fontweight='bold', color='green', transform=ax.transAxes)
        ax.axis('off')

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✅ 可視化を保存: {output_file}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='スコア分布可視化ツール - Stage1/Stage2のスコア分布をグラフ化')
    parser.add_argument('json_file', type=str,
                        help='詳細スコアのJSONファイル (e.g., detailed_scores_cable_5shot.json)')
    parser.add_argument('--stage1_threshold', type=float, default=0.16,
                        help='Stage1閾値 (default: 0.16)')
    parser.add_argument('--stage2_threshold', type=float, default=0.12,
                        help='Stage2閾値 (default: 0.12)')
    parser.add_argument('--output', type=str, default=None,
                        help='出力ファイル名（指定しない場合は画面表示）')

    args = parser.parse_args()

    print(f"\n📂 ファイル: {args.json_file}")
    print(f"🎯 現在の閾値: Stage1={args.stage1_threshold}, Stage2={args.stage2_threshold}")

    detailed_scores = load_detailed_scores(args.json_file)
    visualize_scores(detailed_scores, args.stage1_threshold, args.stage2_threshold, args.output)


if __name__ == "__main__":
    main()
