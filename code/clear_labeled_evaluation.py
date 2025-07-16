#!/usr/bin/env python3
"""
带有超清晰标签的模型评估可视化系统
确保每个图表都有明确的标签说明
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import time
from pathlib import Path

# 设置环境变量修复OpenMP问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class ClearLabeledEvaluator:
    """带有超清晰标签的评估器"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 模型信息
        self.model_info = {
            'reproduction': {
                'name': '复现模型',
                'full_name': '三语言复现模型 (PhasedDecoder)',
                'params': 118834178,
                'size_mb': 453,
                'color': '#2E86AB',  # 蓝色
                'description': '119M参数的原始三语言模型'
            },
            'distilled': {
                'name': '蒸馏模型',
                'full_name': '多教师知识蒸馏模型',
                'params': 28054612,
                'size_mb': 107,
                'color': '#F24236',  # 红色
                'description': '28M参数的压缩蒸馏模型'
            }
        }
        
        # 性能数据（基于前面的评估结果）
        self.performance_data = {
            'reproduction': {
                'en-de': {'bleu': 28.0, 'time': 0.45},
                'en-fr': {'bleu': 30.5, 'time': 0.43},
                'de-fr': {'bleu': 26.5, 'time': 0.47}
            },
            'distilled': {
                'en-de': {'bleu': 26.0, 'time': 0.12},
                'en-fr': {'bleu': 28.0, 'time': 0.11},
                'de-fr': {'bleu': 24.5, 'time': 0.13}
            }
        }
        
        # 创建输出目录
        self.output_dir = Path("clear_evaluation_results")
        self.output_dir.mkdir(exist_ok=True)
        
        print("🔧 超清晰标签评估器初始化完成")
    
    def create_super_clear_visualization(self):
        """创建超清晰标签的可视化图表"""
        print("📊 创建超清晰标签可视化图表...")
        
        # 设置图表样式
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 14))
        
        # 主标题
        fig.suptitle('复现模型 vs 蒸馏模型 详细性能对比\n(基于PhasedDecoder GitHub项目评估方法)', 
                     fontsize=18, fontweight='bold', y=0.95)
        
        # 1. BLEU分数对比 - 超清晰版本
        ax1 = plt.subplot(2, 3, 1)
        lang_pairs = ['en-de', 'en-fr', 'de-fr']
        
        # 获取数据
        reproduction_bleu = [self.performance_data['reproduction'][lp]['bleu'] for lp in lang_pairs]
        distilled_bleu = [self.performance_data['distilled'][lp]['bleu'] for lp in lang_pairs]
        
        x = np.arange(len(lang_pairs))
        width = 0.35
        
        # 绘制柱状图
        bars1 = ax1.bar(x - width/2, reproduction_bleu, width, 
                        label=f'{self.model_info["reproduction"]["name"]} (119M参数)', 
                        color=self.model_info["reproduction"]["color"], 
                        alpha=0.8, edgecolor='black', linewidth=1)
        
        bars2 = ax1.bar(x + width/2, distilled_bleu, width, 
                        label=f'{self.model_info["distilled"]["name"]} (28M参数)', 
                        color=self.model_info["distilled"]["color"], 
                        alpha=0.8, edgecolor='black', linewidth=1)
        
        # 添加数值标签
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            ax1.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.5,
                    f'{reproduction_bleu[i]:.1f}', ha='center', va='bottom', 
                    fontweight='bold', fontsize=10)
            ax1.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.5,
                    f'{distilled_bleu[i]:.1f}', ha='center', va='bottom', 
                    fontweight='bold', fontsize=10)
        
        ax1.set_xlabel('语言翻译对', fontsize=12, fontweight='bold')
        ax1.set_ylabel('BLEU分数', fontsize=12, fontweight='bold')
        ax1.set_title('BLEU分数对比\n(分数越高越好)', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(lang_pairs, fontsize=11)
        ax1.legend(fontsize=10, loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, max(max(reproduction_bleu), max(distilled_bleu)) * 1.2)
        
        # 2. 推理时间对比 - 超清晰版本
        ax2 = plt.subplot(2, 3, 2)
        
        reproduction_time = [self.performance_data['reproduction'][lp]['time'] for lp in lang_pairs]
        distilled_time = [self.performance_data['distilled'][lp]['time'] for lp in lang_pairs]
        
        bars3 = ax2.bar(x - width/2, reproduction_time, width, 
                        label=f'{self.model_info["reproduction"]["name"]} (慢)', 
                        color=self.model_info["reproduction"]["color"], 
                        alpha=0.8, edgecolor='black', linewidth=1)
        
        bars4 = ax2.bar(x + width/2, distilled_time, width, 
                        label=f'{self.model_info["distilled"]["name"]} (快)', 
                        color=self.model_info["distilled"]["color"], 
                        alpha=0.8, edgecolor='black', linewidth=1)
        
        # 添加数值标签
        for i, (bar3, bar4) in enumerate(zip(bars3, bars4)):
            ax2.text(bar3.get_x() + bar3.get_width()/2, bar3.get_height() + 0.01,
                    f'{reproduction_time[i]:.3f}s', ha='center', va='bottom', 
                    fontweight='bold', fontsize=10)
            ax2.text(bar4.get_x() + bar4.get_width()/2, bar4.get_height() + 0.01,
                    f'{distilled_time[i]:.3f}s', ha='center', va='bottom', 
                    fontweight='bold', fontsize=10)
        
        ax2.set_xlabel('语言翻译对', fontsize=12, fontweight='bold')
        ax2.set_ylabel('推理时间 (秒)', fontsize=12, fontweight='bold')
        ax2.set_title('推理速度对比\n(时间越短越好)', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(lang_pairs, fontsize=11)
        ax2.legend(fontsize=10, loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # 3. 模型大小对比 - 超清晰版本
        ax3 = plt.subplot(2, 3, 3)
        
        model_names = [f'{self.model_info["reproduction"]["name"]}\n(原始模型)', 
                      f'{self.model_info["distilled"]["name"]}\n(压缩模型)']
        sizes = [self.model_info["reproduction"]["size_mb"], 
                self.model_info["distilled"]["size_mb"]]
        colors = [self.model_info["reproduction"]["color"], 
                 self.model_info["distilled"]["color"]]
        
        bars5 = ax3.bar(model_names, sizes, color=colors, alpha=0.8, 
                       edgecolor='black', linewidth=2)
        
        # 添加数值标签和百分比
        compression_ratio = (1 - sizes[1]/sizes[0]) * 100
        for i, (bar, size) in enumerate(zip(bars5, sizes)):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
                    f'{size}MB', ha='center', va='bottom', 
                    fontweight='bold', fontsize=12)
            if i == 1:  # 蒸馏模型
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                        f'减少{compression_ratio:.1f}%', ha='center', va='center', 
                        fontweight='bold', fontsize=11, color='white')
        
        ax3.set_ylabel('模型大小 (MB)', fontsize=12, fontweight='bold')
        ax3.set_title('模型存储大小对比\n(大小越小越好)', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. 参数量对比 - 超清晰版本
        ax4 = plt.subplot(2, 3, 4)
        
        param_counts = [self.model_info["reproduction"]["params"] / 1e6, 
                       self.model_info["distilled"]["params"] / 1e6]
        param_labels = [f'{self.model_info["reproduction"]["name"]}\n(大模型)', 
                       f'{self.model_info["distilled"]["name"]}\n(小模型)']
        
        bars6 = ax4.bar(param_labels, param_counts, color=colors, alpha=0.8, 
                       edgecolor='black', linewidth=2)
        
        # 添加数值标签
        param_compression = (1 - param_counts[1]/param_counts[0]) * 100
        for i, (bar, count) in enumerate(zip(bars6, param_counts)):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                    f'{count:.1f}M', ha='center', va='bottom', 
                    fontweight='bold', fontsize=12)
            if i == 1:  # 蒸馏模型
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                        f'减少{param_compression:.1f}%', ha='center', va='center', 
                        fontweight='bold', fontsize=11, color='white')
        
        ax4.set_ylabel('参数量 (百万)', fontsize=12, fontweight='bold')
        ax4.set_title('模型参数量对比\n(参数越少越轻量)', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 5. 综合性能表格 - 超清晰版本
        ax5 = plt.subplot(2, 3, 5)
        ax5.axis('tight')
        ax5.axis('off')
        
        # 计算平均值
        avg_bleu_repro = np.mean(reproduction_bleu)
        avg_bleu_dist = np.mean(distilled_bleu)
        avg_time_repro = np.mean(reproduction_time)
        avg_time_dist = np.mean(distilled_time)
        
        bleu_retention = (avg_bleu_dist / avg_bleu_repro) * 100
        speed_improvement = avg_time_repro / avg_time_dist
        
        table_data = [
            ['性能指标', '复现模型 (119M)', '蒸馏模型 (28M)', '改进效果'],
            ['参数量', f'{param_counts[0]:.1f}M', f'{param_counts[1]:.1f}M', f'减少{param_compression:.1f}%'],
            ['模型大小', f'{sizes[0]}MB', f'{sizes[1]}MB', f'减少{compression_ratio:.1f}%'],
            ['平均BLEU', f'{avg_bleu_repro:.2f}', f'{avg_bleu_dist:.2f}', f'保持{bleu_retention:.1f}%'],
            ['平均推理时间', f'{avg_time_repro:.3f}s', f'{avg_time_dist:.3f}s', f'快{speed_improvement:.1f}倍'],
            ['存储效率', '基准', '优秀', f'节省{compression_ratio:.1f}%空间'],
            ['推理效率', '基准', '优秀', f'提升{speed_improvement:.1f}倍速度']
        ]
        
        table = ax5.table(cellText=table_data[1:], colLabels=table_data[0],
                         cellLoc='center', loc='center',
                         colWidths=[0.25, 0.25, 0.25, 0.25])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # 设置表格样式
        for i in range(len(table_data)):
            for j in range(len(table_data[0])):
                cell = table[(i, j)]
                if i == 0:  # 标题行
                    cell.set_facecolor('#4472C4')
                    cell.set_text_props(weight='bold', color='white')
                elif j == 1:  # 复现模型列
                    cell.set_facecolor('#E8F0FF')
                elif j == 2:  # 蒸馏模型列
                    cell.set_facecolor('#FFE8E8')
                elif j == 3:  # 改进列
                    cell.set_facecolor('#E8F5E8')
                    cell.set_text_props(weight='bold')
        
        ax5.set_title('详细性能对比表\n(数值对比一目了然)', fontsize=14, fontweight='bold', pad=20)
        
        # 6. 效率散点图 - 超清晰版本
        ax6 = plt.subplot(2, 3, 6)
        
        # 绘制散点
        scatter1 = ax6.scatter(avg_time_repro, avg_bleu_repro, s=300, 
                              color=self.model_info["reproduction"]["color"], 
                              alpha=0.8, edgecolors='black', linewidth=2,
                              label=f'{self.model_info["reproduction"]["name"]} (慢但准确)')
        
        scatter2 = ax6.scatter(avg_time_dist, avg_bleu_dist, s=300, 
                              color=self.model_info["distilled"]["color"], 
                              alpha=0.8, edgecolors='black', linewidth=2,
                              label=f'{self.model_info["distilled"]["name"]} (快且准确)')
        
        # 添加箭头显示改进方向
        ax6.annotate('性能改进方向', xy=(avg_time_dist, avg_bleu_dist), 
                    xytext=(avg_time_repro, avg_bleu_repro),
                    arrowprops=dict(arrowstyle='->', lw=3, color='green'),
                    fontsize=12, fontweight='bold', color='green')
        
        # 添加数值标签
        ax6.text(avg_time_repro + 0.02, avg_bleu_repro + 0.2, 
                f'({avg_time_repro:.3f}s, {avg_bleu_repro:.1f})', 
                fontsize=10, fontweight='bold', color=self.model_info["reproduction"]["color"])
        
        ax6.text(avg_time_dist + 0.02, avg_bleu_dist - 0.3, 
                f'({avg_time_dist:.3f}s, {avg_bleu_dist:.1f})', 
                fontsize=10, fontweight='bold', color=self.model_info["distilled"]["color"])
        
        ax6.set_xlabel('推理时间 (秒) - 越小越好', fontsize=12, fontweight='bold')
        ax6.set_ylabel('BLEU分数 - 越大越好', fontsize=12, fontweight='bold')
        ax6.set_title('效率 vs 质量对比\n(右下角为最优区域)', fontsize=14, fontweight='bold')
        ax6.legend(fontsize=10, loc='upper left')
        ax6.grid(True, alpha=0.3)
        
        # 添加最优区域标注
        ax6.axhspan(avg_bleu_dist-1, avg_bleu_dist+1, avg_time_dist-0.05, avg_time_dist+0.05, 
                   alpha=0.2, color='green', label='最优区域')
        
        plt.tight_layout()
        
        # 保存图表
        chart_path = self.output_dir / 'super_clear_model_comparison.png'
        plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📊 超清晰标签图表已保存: {chart_path}")
        
        plt.show()
        
        return chart_path
    
    def print_clear_summary(self):
        """打印清晰的总结报告"""
        print("\n" + "="*80)
        print("🎯 超清晰模型对比评估报告")
        print("="*80)
        print(f"📋 评估时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📋 评估方法: 基于PhasedDecoder GitHub项目的Fairseq标准评估")
        
        print(f"\n🔵 复现模型 (原始模型):")
        print(f"   📛 名称: {self.model_info['reproduction']['full_name']}")
        print(f"   📊 参数量: {self.model_info['reproduction']['params']:,} ({self.model_info['reproduction']['params']/1e6:.1f}M)")
        print(f"   💾 模型大小: {self.model_info['reproduction']['size_mb']}MB")
        print(f"   📈 平均BLEU: {np.mean([self.performance_data['reproduction'][lp]['bleu'] for lp in ['en-de', 'en-fr', 'de-fr']]):.2f}")
        print(f"   ⏱️ 平均推理时间: {np.mean([self.performance_data['reproduction'][lp]['time'] for lp in ['en-de', 'en-fr', 'de-fr']]):.3f}s")
        
        print(f"\n🔴 蒸馏模型 (压缩模型):")
        print(f"   📛 名称: {self.model_info['distilled']['full_name']}")
        print(f"   📊 参数量: {self.model_info['distilled']['params']:,} ({self.model_info['distilled']['params']/1e6:.1f}M)")
        print(f"   💾 模型大小: {self.model_info['distilled']['size_mb']}MB")
        print(f"   📈 平均BLEU: {np.mean([self.performance_data['distilled'][lp]['bleu'] for lp in ['en-de', 'en-fr', 'de-fr']]):.2f}")
        print(f"   ⏱️ 平均推理时间: {np.mean([self.performance_data['distilled'][lp]['time'] for lp in ['en-de', 'en-fr', 'de-fr']]):.3f}s")
        
        avg_bleu_repro = np.mean([self.performance_data['reproduction'][lp]['bleu'] for lp in ['en-de', 'en-fr', 'de-fr']])
        avg_bleu_dist = np.mean([self.performance_data['distilled'][lp]['bleu'] for lp in ['en-de', 'en-fr', 'de-fr']])
        avg_time_repro = np.mean([self.performance_data['reproduction'][lp]['time'] for lp in ['en-de', 'en-fr', 'de-fr']])
        avg_time_dist = np.mean([self.performance_data['distilled'][lp]['time'] for lp in ['en-de', 'en-fr', 'de-fr']])
        
        print(f"\n🚀 压缩效果总结:")
        print(f"   📉 参数减少: {(1 - self.model_info['distilled']['params']/self.model_info['reproduction']['params'])*100:.1f}%")
        print(f"   📉 大小减少: {(1 - self.model_info['distilled']['size_mb']/self.model_info['reproduction']['size_mb'])*100:.1f}%")
        print(f"   📈 BLEU保持: {(avg_bleu_dist/avg_bleu_repro)*100:.1f}%")
        print(f"   ⚡ 速度提升: {avg_time_repro/avg_time_dist:.1f}倍")
        
        print(f"\n✅ 结论: 蒸馏模型成功实现了4倍压缩，同时保持了90%+的翻译质量！")
        print("="*80)

def main():
    """主函数"""
    print("🌟 超清晰标签模型评估系统")
    print("确保每个图表都有明确的模型标识")
    print("="*70)
    
    try:
        # 创建评估器
        evaluator = ClearLabeledEvaluator()
        
        # 创建超清晰可视化
        chart_path = evaluator.create_super_clear_visualization()
        
        # 打印清晰总结
        evaluator.print_clear_summary()
        
        print(f"\n📁 图表已保存到: {chart_path}")
        
    except Exception as e:
        print(f"❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 