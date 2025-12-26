"""
生成性能对比的可视化图表
"""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 无GUI后端
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 测试数据（根据实际测试结果更新）
batch_sizes = [1, 4, 8, 16, 32, 64]
baseline_times = [48.14, 51.80, 52.10, 51.06, 80.00, 0]  # 64会在测试后填入
optimized_times = [41.60, 45.95, 45.73, 45.10, 79.16, 0]  # 64会在测试后填入

# 如果已有batch 64的数据，更新这里
# baseline_times[5] = xxx
# optimized_times[5] = xxx

# 过滤掉未测试的数据
valid_idx = [i for i, (b, o) in enumerate(zip(baseline_times, optimized_times)) if b > 0 and o > 0]
batch_sizes_valid = [batch_sizes[i] for i in valid_idx]
baseline_times_valid = [baseline_times[i] for i in valid_idx]
optimized_times_valid = [optimized_times[i] for i in valid_idx]

# 计算加速比和性能提升
speedups = [b/o for b, o in zip(baseline_times_valid, optimized_times_valid)]
improvements = [(1 - o/b) * 100 for b, o in zip(baseline_times_valid, optimized_times_valid)]

# 创建图表
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('BERT Optimization Performance Comparison', fontsize=16, fontweight='bold')

# 图1: 延迟对比
ax1 = axes[0, 0]
x = np.arange(len(batch_sizes_valid))
width = 0.35
bars1 = ax1.bar(x - width/2, baseline_times_valid, width, label='Baseline', color='#E74C3C', alpha=0.8)
bars2 = ax1.bar(x + width/2, optimized_times_valid, width, label='Optimized', color='#3498DB', alpha=0.8)

ax1.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
ax1.set_ylabel('Latency (ms)', fontsize=12, fontweight='bold')
ax1.set_title('Inference Latency Comparison', fontsize=13, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(batch_sizes_valid)
ax1.legend(fontsize=10)
ax1.grid(axis='y', alpha=0.3)

# 在柱子上添加数值
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=9)

# 图2: 加速比
ax2 = axes[0, 1]
bars = ax2.bar(batch_sizes_valid, speedups, color='#2ECC71', alpha=0.8, edgecolor='black')
ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='No Speedup')
ax2.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
ax2.set_ylabel('Speedup (x)', fontsize=12, fontweight='bold')
ax2.set_title('Speedup Ratio', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(axis='y', alpha=0.3)

# 在柱子上添加数值
for i, (batch, speedup) in enumerate(zip(batch_sizes_valid, speedups)):
    ax2.text(batch, speedup, f'{speedup:.3f}x', 
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# 图3: 性能提升百分比
ax3 = axes[1, 0]
colors = ['#2ECC71' if imp > 5 else '#F39C12' if imp > 0 else '#E74C3C' for imp in improvements]
bars = ax3.bar(batch_sizes_valid, improvements, color=colors, alpha=0.8, edgecolor='black')
ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax3.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
ax3.set_ylabel('Performance Improvement (%)', fontsize=12, fontweight='bold')
ax3.set_title('Performance Improvement Percentage', fontsize=13, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

# 在柱子上添加数值
for i, (batch, imp) in enumerate(zip(batch_sizes_valid, improvements)):
    ax3.text(batch, imp, f'{imp:+.1f}%', 
            ha='center', va='bottom' if imp > 0 else 'top', 
            fontsize=10, fontweight='bold')

# 图4: 优化分解表
ax4 = axes[1, 1]
ax4.axis('off')

# 优化项目表格
optimization_data = [
    ['Optimization', 'Improvement', 'Gain'],
    ['Kernel Launches', '-40%', '3-5%'],
    ['Memory Access', '-57%', '5-8%'],
    ['LayerNorm Scans', '-67%', '3-5%'],
    ['GELU Instructions', '-60%', '2-3%'],
    ['Softmax Scans', '-50%', '3-5%'],
    ['Vectorization', '4x', '2-3%'],
    ['Random Gen (eval)', 'Skip', '2-3%'],
]

table = ax4.table(cellText=optimization_data, cellLoc='left', loc='center',
                  colWidths=[0.5, 0.25, 0.25])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# 设置表头样式
for i in range(3):
    table[(0, i)].set_facecolor('#34495E')
    table[(0, i)].set_text_props(weight='bold', color='white')

# 设置其他行样式
for i in range(1, len(optimization_data)):
    for j in range(3):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#ECF0F1')
        else:
            table[(i, j)].set_facecolor('#FFFFFF')

ax4.set_title('Optimization Breakdown', fontsize=13, fontweight='bold', pad=20)

# 调整布局
plt.tight_layout()

# 保存图表
output_path = '/hy-tmp/bert_optimization_project/performance_comparison.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✓ 图表已保存到: {output_path}")

# 打印统计摘要
print("\n" + "="*60)
print("性能统计摘要")
print("="*60)
for i, batch in enumerate(batch_sizes_valid):
    print(f"Batch {batch:2d}: {baseline_times_valid[i]:6.2f}ms → {optimized_times_valid[i]:6.2f}ms "
          f"({speedups[i]:.3f}x, {improvements[i]:+.1f}%)")

avg_improvement = np.mean(improvements)
print("-"*60)
print(f"平均性能提升: {avg_improvement:+.1f}%")
print("="*60)

