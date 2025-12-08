#!/usr/bin/env python3
"""
2段階異常検知の基本動作テスト
"""

# import numpy as np  # 不要

def two_stage_detection(patch_score, stats_score, object_name, stage1_threshold=0.16, stage2_threshold=0.12):
    """
    2段階異常検知（シンプル版テスト用）
    """
    # Stage 1: パッチベース検知
    if patch_score > stage1_threshold:
        return True, patch_score, "patch_based"
    
    # Stage 2: 統計量による補完検知
    elif stats_score > stage2_threshold:
        return True, stats_score, "statistics_based"
    
    # 正常判定
    else:
        return False, max(patch_score, stats_score), "normal"

def test_two_stage_detection():
    """2段階検知のテストケース（シンプル版）"""
    
    print("🧪 2段階異常検知 - 動作テスト（シンプル版）")
    print("=" * 50)
    
    # Test Case 1: Stage1 検知
    result = two_stage_detection(0.20, 0.05, 'cable', stage1_threshold=0.16, stage2_threshold=0.12)
    print(f"Test 1 (Stage1 detection): {result}")
    assert result[0] == True and result[2] == "patch_based"
    
    # Test Case 2: Stage2 補完検知
    result = two_stage_detection(0.10, 0.15, 'cable', stage1_threshold=0.16, stage2_threshold=0.12)
    print(f"Test 2 (Stage2 detection): {result}")
    assert result[0] == True and result[2] == "statistics_based"
    
    # Test Case 3: 正常判定
    result = two_stage_detection(0.10, 0.08, 'screw', stage1_threshold=0.16, stage2_threshold=0.12)
    print(f"Test 3 (normal): {result}")
    assert result[0] == False and result[2] == "normal"
    
    # Test Case 4: カスタム閾値（積極的検知）
    result = two_stage_detection(0.12, 0.09, 'cable', stage1_threshold=0.16, stage2_threshold=0.08)
    print(f"Test 4 (aggressive Stage2): {result}")
    assert result[0] == True and result[2] == "statistics_based"
    
    # Test Case 5: カスタム閾値（保守的検知）
    result = two_stage_detection(0.12, 0.15, 'pill', stage1_threshold=0.16, stage2_threshold=0.20)
    print(f"Test 5 (conservative): {result}")
    assert result[0] == False and result[2] == "normal"
    
    print("\n✅ 全てのテストケース成功！")
    print("\n📊 期待される効果:")
    print("  • Stage2閾値を下げることで構造的異常（cable_swap等）を検知")
    print("  • シンプルな統一閾値で全カテゴリに適用")
    print("  • Stage2閾値: 0.08-0.10で積極的, 0.12で標準, 0.15-0.20で保守的")

if __name__ == "__main__":
    test_two_stage_detection()