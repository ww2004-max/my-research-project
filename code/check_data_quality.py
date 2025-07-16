#!/usr/bin/env python3
"""
检查训练数据质量
"""

import os
import random

def check_data_samples():
    """检查数据样本"""
    print("🔍 检查Europarl训练数据质量")
    print("=" * 60)
    
    # 检查英语和德语数据
    en_file = "fairseq/models/ZeroTrans/europarl_scripts/mmcr4nlp/europarl/21lingual/train.21langmultiway.europarl-v7.de-en.en"
    de_file = "fairseq/models/ZeroTrans/europarl_scripts/mmcr4nlp/europarl/21lingual/train.21langmultiway.europarl-v7.de-en.de"
    
    if not os.path.exists(en_file):
        print(f"❌ 英语文件不存在: {en_file}")
        return
    
    if not os.path.exists(de_file):
        print(f"❌ 德语文件不存在: {de_file}")
        return
    
    try:
        # 读取英语数据
        print("📖 读取英语训练数据...")
        with open(en_file, 'r', encoding='utf-8') as f:
            en_lines = f.readlines()
        
        # 读取德语数据
        print("📖 读取德语训练数据...")
        with open(de_file, 'r', encoding='utf-8') as f:
            de_lines = f.readlines()
        
        print(f"✅ 数据加载成功")
        print(f"📊 英语句子数量: {len(en_lines):,}")
        print(f"📊 德语句子数量: {len(de_lines):,}")
        
        if len(en_lines) != len(de_lines):
            print("⚠️  警告: 英语和德语句子数量不匹配!")
        
        # 显示前10个样本
        print(f"\n📝 前10个训练样本:")
        for i in range(min(10, len(en_lines))):
            en_text = en_lines[i].strip()
            de_text = de_lines[i].strip() if i < len(de_lines) else "[缺失]"
            
            print(f"\n【样本 {i+1}】")
            print(f"🇬🇧 EN: {en_text}")
            print(f"🇩🇪 DE: {de_text}")
        
        # 随机显示10个样本
        print(f"\n🎲 随机10个训练样本:")
        random_indices = random.sample(range(len(en_lines)), min(10, len(en_lines)))
        
        for i, idx in enumerate(random_indices):
            en_text = en_lines[idx].strip()
            de_text = de_lines[idx].strip() if idx < len(de_lines) else "[缺失]"
            
            print(f"\n【随机样本 {i+1} (行{idx+1})】")
            print(f"🇬🇧 EN: {en_text}")
            print(f"🇩🇪 DE: {de_text}")
        
        # 统计分析
        print(f"\n📊 数据统计分析:")
        
        # 句子长度统计
        en_lengths = [len(line.split()) for line in en_lines]
        de_lengths = [len(line.split()) for line in de_lines]
        
        print(f"英语句子平均长度: {sum(en_lengths)/len(en_lengths):.1f} 词")
        print(f"德语句子平均长度: {sum(de_lengths)/len(de_lengths):.1f} 词")
        print(f"英语最长句子: {max(en_lengths)} 词")
        print(f"德语最长句子: {max(de_lengths)} 词")
        print(f"英语最短句子: {min(en_lengths)} 词")
        print(f"德语最短句子: {min(de_lengths)} 词")
        
        # 检查空行
        empty_en = sum(1 for line in en_lines if not line.strip())
        empty_de = sum(1 for line in de_lines if not line.strip())
        print(f"英语空行数量: {empty_en}")
        print(f"德语空行数量: {empty_de}")
        
        # 检查常见词汇
        print(f"\n🔍 词汇分析:")
        
        # 英语常见词
        en_words = []
        for line in en_lines[:1000]:  # 只分析前1000行
            en_words.extend(line.lower().split())
        
        from collections import Counter
        en_counter = Counter(en_words)
        print(f"英语前20个高频词:")
        for word, count in en_counter.most_common(20):
            print(f"  '{word}': {count}")
        
        # 检查专有名词
        print(f"\n🏛️ 专有名词检查:")
        problem_words = ['Qatar', 'Trautmann', 'Laperrouze', 'Nicaragua']
        for word in problem_words:
            en_count = sum(1 for line in en_lines if word in line)
            de_count = sum(1 for line in de_lines if word in line)
            print(f"  '{word}': EN={en_count}, DE={de_count}")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据检查失败: {e}")
        return False

def check_processed_data():
    """检查处理后的数据"""
    print(f"\n🔧 检查处理后的数据")
    print("=" * 60)
    
    # 检查BPE处理后的数据
    bpe_dir = "fairseq/models/ZeroTrans/europarl_scripts/build_data/bpe"
    
    if os.path.exists(bpe_dir):
        print(f"📂 BPE数据目录存在: {bpe_dir}")
        
        # 列出BPE文件
        bpe_files = os.listdir(bpe_dir)
        print(f"📄 BPE文件数量: {len(bpe_files)}")
        
        # 检查en-de语言对
        en_de_files = [f for f in bpe_files if 'en_de' in f]
        print(f"📄 en-de相关文件: {en_de_files}")
        
        # 检查一个BPE文件的内容
        if en_de_files:
            sample_file = os.path.join(bpe_dir, en_de_files[0])
            try:
                with open(sample_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                print(f"\n📝 BPE样本 ({en_de_files[0]}):")
                for i, line in enumerate(lines[:5]):
                    print(f"  {i+1}: {line.strip()}")
                    
            except Exception as e:
                print(f"❌ 读取BPE文件失败: {e}")
    else:
        print(f"❌ BPE数据目录不存在: {bpe_dir}")

def main():
    print("🔍 Europarl数据集质量检查")
    print("=" * 60)
    
    # 检查原始数据
    if check_data_samples():
        print(f"\n" + "="*60)
        
        # 检查处理后的数据
        check_processed_data()
        
        print(f"\n💡 数据质量评估:")
        print("1. 如果看到大量专有名词重复，说明数据不平衡")
        print("2. 如果句子都很相似，说明数据多样性不足")
        print("3. 如果有很多空行或格式问题，说明数据预处理有问题")
        print("4. Europarl数据本身就是议会发言，专业性很强")
        
        print(f"\n🎯 建议:")
        print("- 如果数据质量有问题，考虑使用其他数据集")
        print("- 如果数据过于专业化，这解释了为什么日常翻译效果不好")
        print("- 可以考虑混合其他更通用的翻译数据集")

if __name__ == "__main__":
    main() 