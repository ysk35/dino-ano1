#!/usr/bin/env python3
"""
実験デバッグスクリプト

Colabで実行して、各実験の設定とログを確認する。

使い方:
    python debug_experiments.py /content/exp_baseline /content/exp_cls /content/exp_multiscale
"""

import os
import sys
import yaml
import json
from pathlib import Path


def debug_experiment(exp_dir):
    """実験ディレクトリを調査"""
    print(f"\n{'='*60}")
    print(f"📁 実験: {exp_dir}")
    print('='*60)

    if not os.path.exists(exp_dir):
        print("  ❌ ディレクトリが存在しません")
        return

    # args.yaml を確認
    args_file = os.path.join(exp_dir, 'args.yaml')
    if os.path.exists(args_file):
        print("\n📋 args.yaml:")
        with open(args_file, 'r') as f:
            args = yaml.safe_load(f)

        # 重要な設定を表示
        key_settings = [
            'use_cls_token', 'cls_weight',
            'use_multiscale', 'layers', 'layer_weights',
            'model_name', 'resolution', 'preprocess', 'shots'
        ]

        for key in key_settings:
            if key in args:
                value = args[key]
                # ハイライト
                if key in ['use_cls_token', 'use_multiscale'] and value:
                    print(f"  ✅ {key}: {value}")
                elif key in ['layers', 'layer_weights', 'cls_weight'] and value:
                    print(f"  ✅ {key}: {value}")
                else:
                    print(f"     {key}: {value}")
    else:
        print("  ⚠️ args.yaml が見つかりません")

    # metrics を確認
    metrics_files = list(Path(exp_dir).glob('**/metrics*.json'))
    if metrics_files:
        print(f"\n📊 メトリクス ({len(metrics_files)}ファイル):")
        for mf in metrics_files[:3]:
            with open(mf, 'r') as f:
                metrics = json.load(f)
            auroc = metrics.get('mean_classification_au_roc', 0) * 100
            f1 = metrics.get('mean_classification_f1', 0) * 100
            print(f"  {mf.name}: AUROC={auroc:.2f}%, F1={f1:.2f}%")
    else:
        print("  ⚠️ メトリクスファイルが見つかりません")

    # ログ確認（もしあれば）
    log_patterns = ['*.log', 'output.txt', 'stdout.txt']
    for pattern in log_patterns:
        logs = list(Path(exp_dir).glob(f'**/{pattern}'))
        if logs:
            print(f"\n📝 ログファイル: {logs[0]}")
            with open(logs[0], 'r') as f:
                content = f.read()

            # 重要なログを検索
            important_phrases = [
                'Multiscale mode',
                'CLS token memory bank',
                'use_cls_token',
                'use_multiscale'
            ]
            for phrase in important_phrases:
                if phrase in content:
                    # その行を表示
                    for line in content.split('\n'):
                        if phrase in line:
                            print(f"  ✅ {line.strip()}")
            break


def compare_settings(exp_dirs):
    """複数実験の設定を比較"""
    print("\n" + "="*60)
    print("🔍 設定比較")
    print("="*60)

    all_args = {}
    for exp_dir in exp_dirs:
        args_file = os.path.join(exp_dir, 'args.yaml')
        if os.path.exists(args_file):
            with open(args_file, 'r') as f:
                all_args[exp_dir] = yaml.safe_load(f)

    if len(all_args) < 2:
        print("比較するための十分な実験がありません")
        return

    # 異なる設定を見つける
    keys_to_check = [
        'use_cls_token', 'cls_weight',
        'use_multiscale', 'layers', 'layer_weights'
    ]

    print("\n重要設定の比較:")
    print("-" * 60)
    header = f"{'設定':<20}"
    for exp in exp_dirs:
        name = os.path.basename(exp)[:15]
        header += f" | {name:>12}"
    print(header)
    print("-" * 60)

    identical_settings = True
    for key in keys_to_check:
        row = f"{key:<20}"
        values = []
        for exp in exp_dirs:
            if exp in all_args:
                val = all_args[exp].get(key, 'N/A')
                values.append(str(val))
                row += f" | {str(val):>12}"
            else:
                values.append('N/A')
                row += f" | {'N/A':>12}"

        # 全て同じかチェック
        unique_values = set(v for v in values if v != 'N/A')
        if len(unique_values) <= 1:
            row += "  ⚠️ 同じ"
            if key in ['use_cls_token', 'use_multiscale']:
                identical_settings = True
        else:
            row += "  ✅ 異なる"
            identical_settings = False

        print(row)

    if identical_settings:
        print("\n" + "⚠️"*20)
        print("警告: 重要設定が全て同じです！")
        print("原因: ブランチ切り替え後にモジュールがリロードされていない可能性")
        print()
        print("解決策:")
        print("  1. Colabランタイムを再起動してから各実験を実行")
        print("  2. または以下のコードを各ブランチ切り替え後に実行:")
        print("     import importlib")
        print("     import src.detection, src.backbones")
        print("     importlib.reload(src.detection)")
        print("     importlib.reload(src.backbones)")
        print("⚠️"*20)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("使い方: python debug_experiments.py <exp_dir1> [exp_dir2] ...")
        print("例: python debug_experiments.py /content/exp_baseline /content/exp_cls")
        sys.exit(1)

    exp_dirs = sys.argv[1:]

    for exp_dir in exp_dirs:
        debug_experiment(exp_dir)

    if len(exp_dirs) > 1:
        compare_settings(exp_dirs)
