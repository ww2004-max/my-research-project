#!/usr/bin/env python3
"""
数据问题总结报告
"""

def main():
    print("🚨 数据集问题诊断报告")
    print("=" * 60)
    
    print("🔍 问题发现:")
    print("1. 训练配置要求6个语言对:")
    print("   - en-de, de-en, en-es, es-en, en-it, it-en")
    
    print("\n2. 实际数据集中只有3个语言对:")
    print("   - de-en ✅ (存在)")
    print("   - es-en ✅ (存在)")  
    print("   - it-en ✅ (存在)")
    print("   - en-de ❌ (缺失)")
    print("   - en-es ❌ (缺失)")
    print("   - en-it ❌ (缺失)")
    
    print("\n📊 影响分析:")
    print("- 训练时只能使用3个语言对，而不是预期的6个")
    print("- 缺失了所有 en→其他语言 的翻译方向")
    print("- 只有 其他语言→en 的翻译方向")
    print("- 这导致模型在英语作为源语言时表现很差")
    
    print("\n🎯 这解释了为什么:")
    print("1. 模型输出奇怪的专有名词")
    print("2. 英语→德语翻译效果很差")
    print("3. 模型过拟合到有限的数据")
    print("4. 继续训练让问题更严重")
    
    print("\n🔧 解决方案:")
    print("方案1: 修改训练配置，只使用存在的语言对")
    print("  --langs de,en,es,it")
    print("  --lang-pairs de-en,es-en,it-en")
    print("  (只支持德语/西语/意语→英语)")
    
    print("\n方案2: 寻找完整的Europarl数据集")
    print("  - 下载完整的Europarl v7数据")
    print("  - 重新预处理数据")
    
    print("\n方案3: 使用其他数据集")
    print("  - WMT翻译数据集")
    print("  - OPUS数据集")
    print("  - 多语言通用数据集")
    
    print("\n💡 推荐:")
    print("1. 立即使用方案1修复训练配置")
    print("2. 重新训练一个只支持3个方向的模型")
    print("3. 长期考虑获取完整数据集")
    
    print("\n🎉 好消息:")
    print("- 训练流程本身是正确的")
    print("- 模型架构没有问题")
    print("- 只是数据不完整导致的问题")
    print("- 修复后应该能得到正常的翻译模型")

if __name__ == "__main__":
    main() 