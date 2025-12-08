#!/usr/bin/env python3
"""
実験結果比較スクリプト

使用方法:
    # ディレクトリを指定して比較
    python compare_experiments.py --dirs exp_baseline exp_cls exp_multiscale

    # Colab上で使用する場合
    from compare_experiments import compare_experiments
    compare_experiments({
        'baseline': '/content/exp_baseline',
        'cls_token': '/content/exp_cls',
        'multiscale': '/content/exp_multiscale'
    })
"""

import json
import os
import argparse
from pathlib import Path
from collections import defaultdict


def find_metrics_file(directory):
    """
    ディレクトリ内のメトリクスファイルを探す
    複数のパターンに対応:
    - metrics_seed=0.json (直接)
    - metrics_seed=*.json (任意のシード)
    - サブディレクトリ内
    """
    directory = Path(directory)

    # パターン1: 直接配下
    for pattern in ['metrics_seed=*.json', 'metrics.json']:
        files = list(directory.glob(pattern))
        if files:
            return files

    # パターン2: 1階層下
    for pattern in ['*/metrics_seed=*.json', '*/metrics.json']:
        files = list(directory.glob(pattern))
        if files:
            return files

    # パターン3: 2階層下 (results_MVTec/model/shot-preprocess/metrics.json)
    for pattern in ['**/metrics_seed=*.json', '**/metrics.json']:
        files = list(directory.glob(pattern))
        if files:
            return files[:5]  # 多すぎる場合は最初の5つ

    return []


def load_metrics(metrics_path):
    """メトリクスファイルを読み込む"""
    with open(metrics_path, 'r') as f:
        return json.load(f)


def compare_experiments(experiment_dirs, show_per_object=False, show_diff=True):
    """
    複数の実験結果を比較

    Args:
        experiment_dirs: dict {実験名: ディレクトリパス}
        show_per_object: オブジェクト別の詳細を表示
        show_diff: ベースラインとの差分を表示
    """
    results = {}
    all_objects = set()

    print("=" * 70)
    print("実験結果の比較")
    print("=" * 70)

    # 各実験のメトリクスを読み込み
    for name, directory in experiment_dirs.items():
        metrics_files = find_metrics_file(directory)

        if not metrics_files:
            print(f"⚠️  {name}: メトリクスファイルが見つかりません ({directory})")
            print(f"   ディレクトリ内容: {list(Path(directory).glob('*'))[:10]}")
            continue

        # 複数シードの場合は平均を取る
        all_metrics = []
        for mf in metrics_files:
            try:
                metrics = load_metrics(mf)
                all_metrics.append(metrics)
                print(f"✓ {name}: {mf}")
            except Exception as e:
                print(f"⚠️  {name}: 読み込みエラー ({mf}): {e}")

        if not all_metrics:
            continue

        # 平均を計算
        averaged = {}
        keys = all_metrics[0].keys()

        for key in keys:
            if key.startswith('mean_'):
                # 数値の平均
                values = [m[key] for m in all_metrics if key in m]
                averaged[key] = sum(values) / len(values) if values else 0
            elif isinstance(all_metrics[0].get(key), dict):
                # オブジェクト別メトリクス
                all_objects.add(key)
                averaged[key] = {}
                for metric_name in all_metrics[0][key].keys():
                    values = [m[key][metric_name] for m in all_metrics if key in m]
                    averaged[key][metric_name] = sum(values) / len(values) if values else 0

        results[name] = averaged

    if not results:
        print("\n❌ 比較可能な結果がありません")
        return None

    # サマリー表示
    print("\n" + "=" * 70)
    print("📊 サマリー (平均値)")
    print("=" * 70)

    # ヘッダー
    exp_names = list(results.keys())
    header = f"{'Metric':<25}"
    for name in exp_names:
        header += f" | {name:>12}"
    if show_diff and len(exp_names) > 1:
        header += f" | {'Diff':>10}"
    print(header)
    print("-" * len(header))

    # 主要メトリクス
    metrics_to_show = [
        ('mean_classification_au_roc', 'AUROC (分類)'),
        ('mean_classification_ap', 'AP (分類)'),
        ('mean_classification_f1', 'F1 (分類)'),
        ('mean_segmentation_au_roc', 'AUROC (セグメント)'),
        ('mean_segmentation_f1', 'F1 (セグメント)'),
        ('mean_au_pro', 'AU-PRO'),
    ]

    baseline_name = exp_names[0] if exp_names else None

    for metric_key, metric_label in metrics_to_show:
        if not any(metric_key in results[name] for name in exp_names):
            continue

        row = f"{metric_label:<25}"
        baseline_val = None

        for name in exp_names:
            val = results[name].get(metric_key, None)
            if val is not None:
                if name == baseline_name:
                    baseline_val = val
                row += f" | {val*100:>11.2f}%"
            else:
                row += f" | {'N/A':>12}"

        # 差分表示
        if show_diff and len(exp_names) > 1 and baseline_val is not None:
            last_val = results[exp_names[-1]].get(metric_key, None)
            if last_val is not None:
                diff = (last_val - baseline_val) * 100
                diff_str = f"+{diff:.2f}%" if diff >= 0 else f"{diff:.2f}%"
                row += f" | {diff_str:>10}"

        print(row)

    # オブジェクト別詳細
    if show_per_object and all_objects:
        print("\n" + "=" * 70)
        print("📋 オブジェクト別 AUROC")
        print("=" * 70)

        # ソート (MVTec順)
        mvtec_order = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
                       "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
                       "transistor", "wood", "zipper"]
        sorted_objects = sorted(all_objects,
                               key=lambda x: mvtec_order.index(x) if x in mvtec_order else 100)

        header = f"{'Object':<15}"
        for name in exp_names:
            header += f" | {name:>12}"
        if show_diff and len(exp_names) > 1:
            header += f" | {'Best':>10}"
        print(header)
        print("-" * len(header))

        for obj in sorted_objects:
            if obj.startswith('mean_'):
                continue

            row = f"{obj:<15}"
            values = []

            for name in exp_names:
                if obj in results[name]:
                    val = results[name][obj].get('classification_AUROC', None)
                    if val is not None:
                        values.append((name, val))
                        row += f" | {val*100:>11.2f}%"
                    else:
                        row += f" | {'N/A':>12}"
                else:
                    row += f" | {'N/A':>12}"

            # 最良のモデルをマーク
            if show_diff and len(values) > 1:
                best_name = max(values, key=lambda x: x[1])[0]
                row += f" | {best_name[:10]:>10}"

            print(row)

    return results


def debug_directory(directory):
    """ディレクトリ構造をデバッグ表示"""
    print(f"\n🔍 ディレクトリ調査: {directory}")
    print("-" * 50)

    if not os.path.exists(directory):
        print(f"  ❌ 存在しません")
        return

    for root, dirs, files in os.walk(directory):
        level = root.replace(directory, '').count(os.sep)
        indent = '  ' * level
        print(f"{indent}{os.path.basename(root)}/")

        # 最大2階層まで
        if level >= 2:
            continue

        subindent = '  ' * (level + 1)
        for file in files[:10]:  # 最大10ファイル
            print(f"{subindent}{file}")
        if len(files) > 10:
            print(f"{subindent}... (and {len(files) - 10} more files)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='実験結果の比較')
    parser.add_argument('--dirs', nargs='+', required=True,
                        help='比較する実験ディレクトリ (例: exp_baseline exp_cls)')
    parser.add_argument('--names', nargs='+', default=None,
                        help='実験の名前 (指定しない場合はディレクトリ名を使用)')
    parser.add_argument('--per-object', action='store_true',
                        help='オブジェクト別の詳細を表示')
    parser.add_argument('--debug', action='store_true',
                        help='ディレクトリ構造をデバッグ表示')

    args = parser.parse_args()

    if args.debug:
        for d in args.dirs:
            debug_directory(d)

    names = args.names if args.names else [os.path.basename(d) for d in args.dirs]
    experiment_dirs = dict(zip(names, args.dirs))

    compare_experiments(experiment_dirs, show_per_object=args.per_object)
