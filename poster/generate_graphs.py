import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.size'] = 14

variants = ['Unfused', 'Partial-ST', 'Partial-MT', 'Fully\nFused']
colors = {
    'unfused': '#4A90D9',
    'partial_st': '#D4A037',
    'partial_mt': '#E67E22',
    'fused': '#27AE60',
    'read': '#4A90D9',
    'write': '#E74C3C',
}
bar_colors = [colors['unfused'], colors['partial_st'], colors['partial_mt'], colors['fused']]

# ============================================================
# GRAPH 1: GB Data Movement (Stacked Bar — Reads + Writes)
# ============================================================
fig, (ax_r, ax_w) = plt.subplots(1, 2, figsize=(12, 5))

gb_reads  = [8768, 8512, 8512, 8256]
gb_writes = [576,  320,  320,  64]

x = np.arange(len(variants))
width = 0.55

# Reads
bars_r = ax_r.bar(x, gb_reads, width, color=bar_colors, edgecolor='white', linewidth=0.5)
for i, v in enumerate(gb_reads):
    ax_r.text(i, v + 150, f'{v:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax_r.set_ylabel('Vector Count (16B each)', fontsize=13)
ax_r.set_title('GB Reads (N=64, d=16)', fontsize=15, fontweight='bold')
ax_r.set_xticks(x)
ax_r.set_xticklabels(variants, fontsize=11)
ax_r.set_ylim(0, 10500)
ax_r.spines['top'].set_visible(False)
ax_r.spines['right'].set_visible(False)

# Writes
bars_w = ax_w.bar(x, gb_writes, width, color=bar_colors, edgecolor='white', linewidth=0.5)
for i, v in enumerate(gb_writes):
    ax_w.text(i, v + 15, f'{v}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax_w.set_ylabel('Vector Count (16B each)', fontsize=13)
ax_w.set_title('GB Writes (N=64, d=16)', fontsize=15, fontweight='bold')
ax_w.set_xticks(x)
ax_w.set_xticklabels(variants, fontsize=11)
ax_w.set_ylim(0, 700)
ax_w.spines['top'].set_visible(False)
ax_w.spines['right'].set_visible(False)

# 9x annotation
ax_w.annotate('', xy=(3, 64), xytext=(0, 576),
            arrowprops=dict(arrowstyle='->', color='#333333', lw=2, connectionstyle='arc3,rad=-0.15'))
ax_w.text(1.5, 450, r'9$\times$ reduction', fontsize=13, fontweight='bold', color='#333333', rotation=-25)

plt.tight_layout()
plt.savefig('figures/data_mvmt.png', dpi=200, bbox_inches='tight')
plt.savefig('figures/data_mvmt.pdf', bbox_inches='tight')
print("Saved figures/data_mvmt.png/pdf")


# ============================================================
# GRAPH 2: Latency Comparison (N=64 only — the interesting case)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

latency_64 = [84296, 130793, 31329, 66454]

bars = ax.bar(x, latency_64, width, color=bar_colors, edgecolor='white', linewidth=0.5)

for i, v in enumerate(latency_64):
    ax.text(i, v + 2500, f'{v:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Speedup annotations
speedups = [1.0, 0.64, 2.69, 1.27]
for i, s in enumerate(speedups):
    if i == 0:
        continue
    color = bar_colors[i]
    label = f'{s:.2f}' + r'$\times$'
    ax.text(i, latency_64[i] + 9000, label, ha='center', fontsize=11, color=color, fontstyle='italic', fontweight='bold')

ax.set_ylabel('Simulation Clocks', fontsize=13)
ax.set_title('End-to-End Latency (N=64, d=16)', fontsize=15, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(variants, fontsize=12)
ax.set_ylim(0, 160000)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('figures/latency.png', dpi=200, bbox_inches='tight')
plt.savefig('figures/latency.pdf', bbox_inches='tight')
print("Saved figures/latency.png/pdf")


# ============================================================
# GRAPH 3: Area vs Speedup Trade-off (Scatter)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5.5))

area_scores = [74977, 80015, 73843, 141900]
speedup_vals = [1.0, 0.64, 2.69, 1.27]
labels = ['Unfused', 'Partial-ST', 'Partial-MT', 'Fully Fused']

for i in range(4):
    ax.scatter(area_scores[i], speedup_vals[i], s=250, c=bar_colors[i], edgecolors='black', linewidth=1.5, zorder=5)

# Labels with offsets to avoid overlap
offsets = [(15, -15), (-50, -18), (15, 10), (15, -15)]
for i in range(4):
    ax.annotate(labels[i], (area_scores[i], speedup_vals[i]),
                textcoords="offset points", xytext=offsets[i],
                fontsize=12, fontweight='bold', ha='left')

ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Unfused baseline')
ax.set_xlabel('HLS Area Score', fontsize=13)
ax.set_ylabel('Speedup vs Unfused', fontsize=13)
ax.set_title('Area vs Performance Trade-off', fontsize=15, fontweight='bold')
ax.set_xlim(50000, 170000)
ax.set_ylim(0.3, 3.2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(fontsize=11)

# Highlight Partial-MT as Pareto-optimal
ax.annotate('Best efficiency:\n2.69x speed, 1.02x area',
            xy=(73843, 2.69), xytext=(100000, 2.9),
            fontsize=10, fontstyle='italic', color=colors['partial_mt'],
            arrowprops=dict(arrowstyle='->', color=colors['partial_mt'], lw=1.5),
            ha='center')

plt.tight_layout()
plt.savefig('figures/area_tradeoff.png', dpi=200, bbox_inches='tight')
plt.savefig('figures/area_tradeoff.pdf', bbox_inches='tight')
print("Saved figures/area_tradeoff.png/pdf")

print("\nAll graphs generated!")
