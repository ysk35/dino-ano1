#!/bin/bash
# 2段階異常検知の閾値調整コマンド例（シンプル版）

echo "🔧 2段階異常検知の閾値調整例（シンプル版）"
echo "================================================"
echo ""

echo "📊 デフォルト閾値:"
echo "Stage1 (パッチベース): 0.16"
echo "Stage2 (統計量ベース): 0.12"
echo ""

echo "🚀 閾値調整コマンド例:"
echo ""

echo "1️⃣ デフォルト実行（baseline測定）:"
echo "python run_anomalydino.py --dataset MVTec --shots 1"
echo ""

echo "2️⃣ Stage2を積極的に（cable_swap等の検知改善）:"
echo "python run_anomalydino.py --dataset MVTec --shots 1 --stage2_threshold 0.08"
echo ""

echo "3️⃣ Stage2をさらに積極的に:"
echo "python run_anomalydino.py --dataset MVTec --shots 1 --stage2_threshold 0.06"
echo ""

echo "4️⃣ Stage2を保守的に（FP抑制）:"
echo "python run_anomalydino.py --dataset MVTec --shots 1 --stage2_threshold 0.15"
echo ""

echo "5️⃣ Stage1も調整:"
echo "python run_anomalydino.py --dataset MVTec --shots 1 --stage1_threshold 0.14 --stage2_threshold 0.08"
echo ""

echo "📈 効果測定方法:"
echo "- 結果ディレクトリの metrics_seed=0.json で性能確認"
echo "- フォルダ別エラー分析で cable_swap, missing_wire の改善確認"
echo "- FP増加をモニタリング"
echo ""

echo "🎯 最適化戦略:"
echo "1. baseline測定: --stage2_threshold 0.12 (デフォルト)"
echo "2. 積極化テスト: --stage2_threshold 0.08"
echo "3. 最適点探索: 0.06, 0.04等を試す"
echo "4. FP増加が許容範囲内か確認"
echo ""

echo "💡 期待値:"
echo "• Stage2閾値 0.08-0.10: cable_swap等の構造的異常を検知"
echo "• 統一閾値なので設定が簡単"
echo "• 失敗している異常を少しでもキャッチできればOK"