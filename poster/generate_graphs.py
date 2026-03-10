import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.size'] = 14

variants = ['Unfused', 'Partial\nFused', 'Fully\nFused']
colors_read = '#4A90D9'
colors_write = '#E74C3C'
colors_latency_16 = '#7FB3D8'
colors_latency_64 = '#2C5F8A'

# ============================================================
# GRAPH 1: GB Data Movement (Stacked Bar — Reads + Writes)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

gb_reads  = [8768, 8512, 8256]
gb_writes = [576,  320,  64]

x = np.arange(len(variants))
width = 0.5

bars_r = ax.bar(x, gb_reads, width, label='GB Reads', color=colors_read, edgecolor='white', linewidth=0.5)
bars_w = ax.bar(x, gb_writes, width, bottom=gb_reads, label='GB Writes', color=colors_write, edgecolor='white', linewidth=0.5)

# Annotate write counts on the write bars
for i, (r, w) in enumerate(zip(gb_reads, gb_writes)):
    ax.text(i, r + w + 150, f'{w} writes', ha='center', va='bottom', fontsize=12, fontweight='bold', color=colors_write)

ax.set_ylabel('Vector Count (16B each)', fontsize=13)
ax.set_title('GB Memory Traffic (N=64, d=16)', fontsize=15, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(variants, fontsize=13)
ax.legend(loc='upper right', fontsize=12)
ax.set_ylim(0, 11000)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add 9x annotation arrow
ax.annotate('9× fewer\nwrites', xy=(2, 8256 + 64 + 150), xytext=(2.45, 9800),
            fontsize=12, fontweight='bold', color=colors_write,
            arrowprops=dict(arrowstyle='->', color=colors_write, lw=1.5),
            ha='center')

plt.tight_layout()
plt.savefig('graph_data_movement.png', dpi=200, bbox_inches='tight')
plt.savefig('graph_data_movement.pdf', bbox_inches='tight')
print("Saved graph_data_movement.png/pdf")


# ============================================================
# GRAPH 2: Latency Comparison (Grouped Bar — N=16 and N=64)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

variants_short = ['Unfused', 'Partial\nFused', 'Fully\nFused']

# N=16
latency_16 = [5672, 8565, 4392]
x = np.arange(len(variants_short))
width = 0.5

bars16 = ax1.bar(x, latency_16, width, color=[colors_latency_64, '#D4A037', '#27AE60'], edgecolor='white', linewidth=0.5)
ax1.set_ylabel('Simulation Clocks', fontsize=13)
ax1.set_title('Latency (N=16)', fontsize=15, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(variants_short, fontsize=12)
ax1.set_ylim(0, 10000)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

for i, v in enumerate(latency_16):
    ax1.text(i, v + 200, f'{v:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Speedup annotations for N=16
ax1.text(2, 4392 + 800, '1.29×', ha='center', fontsize=10, color='#27AE60', fontstyle='italic')

# N=64
latency_64 = [84296, 131561, 66454]

bars64 = ax2.bar(x, latency_64, width, color=[colors_latency_64, '#D4A037', '#27AE60'], edgecolor='white', linewidth=0.5)
ax2.set_ylabel('Simulation Clocks', fontsize=13)
ax2.set_title('Latency (N=64)', fontsize=15, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(variants_short, fontsize=12)
ax2.set_ylim(0, 155000)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

for i, v in enumerate(latency_64):
    ax2.text(i, v + 3000, f'{v:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Speedup annotations for N=64
ax2.text(2, 66454 + 12000, '1.27×', ha='center', fontsize=10, color='#27AE60', fontstyle='italic')
ax2.text(1, 131561 + 12000, '0.64×', ha='center', fontsize=10, color='#D4A037', fontstyle='italic')

plt.tight_layout()
plt.savefig('graph_latency.png', dpi=200, bbox_inches='tight')
plt.savefig('graph_latency.pdf', bbox_inches='tight')
print("Saved graph_latency.png/pdf")


# ============================================================
# GRAPH 3: GB Writes Only (Dramatic reduction bar chart)
# ============================================================
fig, ax = plt.subplots(figsize=(7, 5))

writes = [576, 320, 64]

bars = ax.bar(x, writes, width, color=[colors_latency_64, '#D4A037', '#27AE60'], edgecolor='white', linewidth=0.5)

for i, v in enumerate(writes):
    ax.text(i, v + 15, f'{v}', ha='center', va='bottom', fontsize=13, fontweight='bold')

ax.set_ylabel('GB Write Vectors', fontsize=13)
ax.set_title('SRAM Write Reduction (N=64, d=16)', fontsize=15, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(variants_short, fontsize=13)
ax.set_ylim(0, 700)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Arrow showing 9x reduction
ax.annotate('', xy=(2, 64), xytext=(0, 576),
            arrowprops=dict(arrowstyle='->', color='#333333', lw=2, connectionstyle='arc3,rad=-0.2'))
ax.text(1.3, 400, '9× reduction', fontsize=13, fontweight='bold', color='#333333', rotation=-30)

plt.tight_layout()
plt.savefig('graph_writes_reduction.png', dpi=200, bbox_inches='tight')
plt.savefig('graph_writes_reduction.pdf', bbox_inches='tight')
print("Saved graph_writes_reduction.png/pdf")


# ============================================================
# GRAPH 4: Area vs Speedup Trade-off (Scatter/Bubble)
# ============================================================
fig, ax = plt.subplots(figsize=(7, 5))

area_scores = [72699, 80015, 141879]
speedups = [1.0, 0.64, 1.27]
labels = ['Unfused', 'Partial Fused', 'Fully Fused']
point_colors = [colors_latency_64, '#D4A037', '#27AE60']

for i in range(3):
    ax.scatter(area_scores[i], speedups[i], s=200, c=point_colors[i], edgecolors='black', linewidth=1, zorder=5)
    offset_x = 3000 if i != 1 else -3000
    offset_y = 0.04 if i != 1 else -0.06
    ax.annotate(labels[i], (area_scores[i], speedups[i]),
                textcoords="offset points", xytext=(offset_x//200, offset_y*300),
                fontsize=12, fontweight='bold', ha='center')

ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Unfused baseline')
ax.set_xlabel('HLS Area Score', fontsize=13)
ax.set_ylabel('Speedup vs Unfused', fontsize=13)
ax.set_title('Area vs Performance Trade-off', fontsize=15, fontweight='bold')
ax.set_xlim(50000, 170000)
ax.set_ylim(0.4, 1.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig('graph_area_tradeoff.png', dpi=200, bbox_inches='tight')
plt.savefig('graph_area_tradeoff.pdf', bbox_inches='tight')
print("Saved graph_area_tradeoff.png/pdf")

print("\nAll graphs generated!")
