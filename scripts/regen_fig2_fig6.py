#!/usr/bin/env python3
"""
Regenerate Figure 2 (per-subject bar chart) and Figure 6 (statistical comparison)
at publication quality, saving directly into MANUSCRIPT/LEGEND_FINAL/.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy.stats import wilcoxon
from pathlib import Path

# ── destination ──────────────────────────────────────────────────────────────
OUT = Path('MANUSCRIPT/LEGEND_FINAL')

# ── global typography / style ─────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'legend.fontsize': 8,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '#cccccc',
    'lines.linewidth': 1.2,
    'grid.linewidth': 0.5,
    'grid.alpha': 0.35,
})

# ── palette (colour-blind safe) ───────────────────────────────────────────────
C_EEGNET   = '#4878CF'   # blue
C_SCNN     = '#6ACC65'   # green
C_STAGE1   = '#D65F5F'   # soft red
C_LEGEND   = '#B47CC7'   # purple
RANDOM_RED = '#E8474C'

# ── load verified per-subject accuracies ─────────────────────────────────────
def load_per_subject(path):
    d = json.load(open(path))
    ps = d['per_subject']
    subjects = sorted(ps.keys())
    return subjects, [ps[s] for s in subjects]

subjects, v3_acc    = load_per_subject('results/hybrid_loso_v3/summary.json')
_,        stg1_acc  = load_per_subject('results/hybrid_loso/summary.json')
_,        eeg_acc   = load_per_subject('results/baselines_EEGNet/loso_summary.json')
_,        scnn_acc  = load_per_subject('results/baselines_ShallowConvNet/loso_summary.json')

v3_acc   = np.array(v3_acc)
stg1_acc = np.array(stg1_acc)
eeg_acc  = np.array(eeg_acc)
scnn_acc = np.array(scnn_acc)

# short labels for x-axis
xlabels = [s.replace('NIS', '') for s in subjects]   # 001 … 010

# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 2 — Per-subject grouped bar chart
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 2 …")

n = len(subjects)
x = np.arange(n)
w = 0.19          # bar width; 4 bars + tiny gap per group

fig2, ax = plt.subplots(figsize=(8.5, 4.0))

b1 = ax.bar(x - 1.5*w, eeg_acc,  w, color=C_EEGNET,  label='EEGNet',        zorder=3)
b2 = ax.bar(x - 0.5*w, scnn_acc, w, color=C_SCNN,    label='ShallowConvNet', zorder=3)
b3 = ax.bar(x + 0.5*w, stg1_acc, w, color=C_STAGE1,  label='Stage-1',        zorder=3)
b4 = ax.bar(x + 1.5*w, v3_acc,   w, color=C_LEGEND,  label='LEGEND',         zorder=3, linewidth=0.6, edgecolor='#5a318c')

# random-chance baseline
ax.axhline(25, color=RANDOM_RED, linestyle='--', linewidth=1.2,
           label='Chance (25 %)', zorder=4)

# annotate LEGEND bars with value
for xi, acc in zip(x, v3_acc):
    ax.text(xi + 1.5*w, acc + 1.2, f'{acc:.1f}',
            ha='center', va='bottom', fontsize=6.5, color='#3d1a5e', fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels([f'NIS{lb}' for lb in xlabels], rotation=30, ha='right')
ax.set_ylabel('LOSO Accuracy (%)', fontweight='bold')
ax.set_ylim(0, 105)
ax.yaxis.set_major_locator(plt.MultipleLocator(10))
ax.yaxis.set_minor_locator(plt.MultipleLocator(5))
ax.grid(True, which='major', axis='y', zorder=0)
ax.set_title('Per-subject LOSO accuracy — Steele dataset (4-class motor imagery)',
             pad=7, fontweight='bold')

leg = ax.legend(loc='upper left', ncol=5,
                bbox_to_anchor=(0, 1.0), framealpha=0.9)
leg.get_frame().set_linewidth(0.6)

# mean ± std summary line under chart
summary = (f"Means:  EEGNet {eeg_acc.mean():.2f}±{eeg_acc.std(ddof=1):.2f}   "
           f"ShallowConvNet {scnn_acc.mean():.2f}±{scnn_acc.std(ddof=1):.2f}   "
           f"Stage-1 {stg1_acc.mean():.2f}±{stg1_acc.std(ddof=1):.2f}   "
           f"LEGEND {v3_acc.mean():.2f}±{v3_acc.std(ddof=1):.2f}")
fig2.text(0.5, -0.04, summary, ha='center', fontsize=7, style='italic', color='#444444')

fig2.tight_layout()
fig2.savefig(OUT / 'figure2_subject_heatmaps.png')
fig2.savefig(OUT / 'figure2_subject_heatmaps.pdf')
plt.close(fig2)
print(f"  ✅  Saved figure2_subject_heatmaps.png/.pdf  →  {OUT}")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 6 — Statistical comparison (box + Wilcoxon)
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 6 …")

ALPHA = 0.05

# one-tailed Wilcoxon:  H1: LEGEND > baseline
def wilcox_p(baseline, legend):
    try:
        _, p2 = wilcoxon(legend, baseline, alternative='greater')
        return p2
    except Exception:
        return float('nan')

p_eeg  = wilcox_p(eeg_acc,  v3_acc)
p_scnn = wilcox_p(scnn_acc, v3_acc)
p_stg1 = wilcox_p(stg1_acc, v3_acc)

print(f"  Wilcoxon p-values:  EEGNet={p_eeg:.4f}  ShallowCNN={p_scnn:.4f}  Stage-1={p_stg1:.4f}")

fig6 = plt.figure(figsize=(8.5, 4.2))
gs = GridSpec(1, 2, figure=fig6, width_ratios=[1.7, 1], wspace=0.38)

# ── LEFT: paired box plots with jittered fold dots ────────────────────────────
ax_box = fig6.add_subplot(gs[0])

methods = {
    'EEGNet\n(EEG)':        (eeg_acc,  C_EEGNET),
    'ShallowConvNet\n(EEG)':(scnn_acc, C_SCNN),
    'Stage-1\n(all modal)': (stg1_acc, C_STAGE1),
    'LEGEND\n(all modal)':  (v3_acc,   C_LEGEND),
}

bp_data   = [v for v, _ in methods.values()]
bp_colors = [c for _, c in methods.values()]
bp_labels = list(methods.keys())

bp = ax_box.boxplot(
    bp_data,
    patch_artist=True,
    widths=0.45,
    medianprops=dict(color='black', linewidth=1.8),
    whiskerprops=dict(linewidth=0.9),
    capprops=dict(linewidth=0.9),
    flierprops=dict(marker='', markersize=0),
    zorder=3,
)

for patch, colour in zip(bp['boxes'], bp_colors):
    patch.set_facecolor(colour)
    patch.set_alpha(0.55)

# jittered subject dots + connecting lines (light)
rng = np.random.default_rng(42)
positions = [1, 2, 3, 4]
for i in range(n):
    xs = np.array(positions) + rng.uniform(-0.09, 0.09, len(positions))
    ys = np.array([v[i] for v in bp_data])
    ax_box.plot(xs, ys, color='#bbbbbb', linewidth=0.5, zorder=2, alpha=0.6)
    for xi, yi, colour in zip(xs, ys, bp_colors):
        ax_box.scatter(xi, yi, color=colour, s=16, zorder=4,
                       edgecolors='white', linewidths=0.4)

ax_box.axhline(25, color=RANDOM_RED, linestyle='--', linewidth=1.0,
               label='Chance (25 %)', zorder=5)
ax_box.set_xticks(positions)
ax_box.set_xticklabels(bp_labels, fontsize=8)
ax_box.set_ylabel('LOSO Fold Accuracy (%)', fontweight='bold')
ax_box.set_ylim(5, 110)
ax_box.yaxis.set_major_locator(plt.MultipleLocator(10))
ax_box.grid(True, axis='y', zorder=0)
ax_box.set_title('Accuracy distribution across 10 folds', fontweight='bold', pad=5)
ax_box.legend(loc='upper left', fontsize=7.5)

# significance brackets
def sig_bracket(ax, x1, x2, y, pval):
    h = 1.8
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=0.9, c='black')
    stars = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
    pstr  = f'p={pval:.3f} ({stars})'
    ax.text((x1+x2)/2, y+h+0.3, pstr, ha='center', va='bottom',
            fontsize=7, color='black')

sig_bracket(ax_box, 1, 4, 88,  p_eeg)
sig_bracket(ax_box, 2, 4, 96,  p_scnn)
sig_bracket(ax_box, 3, 4, 104, p_stg1)

# ── RIGHT: p-value bar chart ──────────────────────────────────────────────────
ax_pval = fig6.add_subplot(gs[1])

baselines   = ['EEGNet', 'ShallowConvNet', 'Stage-1']
pvals       = [p_eeg, p_scnn, p_stg1]
bar_colors  = [C_EEGNET, C_SCNN, C_STAGE1]
ypos        = np.arange(len(baselines))

bars = ax_pval.barh(ypos, pvals, color=bar_colors, alpha=0.75,
                    edgecolor='white', linewidth=0.6, height=0.5, zorder=3)
ax_pval.axvline(ALPHA, color=RANDOM_RED, linestyle='--', linewidth=1.2,
                label=f'α = {ALPHA}', zorder=4)

for bar, p in zip(bars, pvals):
    label = f'p = {p:.3f}' if p >= 0.001 else f'p < 0.001'
    ax_pval.text(max(p + 0.003, 0.008), bar.get_y() + bar.get_height()/2,
                 label, va='center', fontsize=8, fontweight='bold')

ax_pval.set_yticks(ypos)
ax_pval.set_yticklabels(baselines, fontsize=8.5)
ax_pval.set_xlabel('Wilcoxon p-value\n(one-tailed, H₁: LEGEND > baseline)', fontweight='bold')
ax_pval.set_xlim(0, max(pvals) * 1.55 + 0.04)
ax_pval.set_title('Significance tests', fontweight='bold', pad=5)
ax_pval.grid(True, axis='x', zorder=0)
ax_pval.legend(loc='lower right', fontsize=7.5)
ax_pval.invert_yaxis()

fig6.suptitle('Statistical comparison of LOSO accuracies — Steele dataset',
              fontweight='bold', y=1.01, fontsize=10)
fig6.tight_layout()
fig6.savefig(OUT / 'figure6_3d_hyperbolic_embedding.pdf')   # keeps original filename
fig6.savefig(OUT / 'figure6_3d_hyperbolic_embedding.png')
plt.close(fig6)
print(f"  ✅  Saved figure6_3d_hyperbolic_embedding.pdf/.png  →  {OUT}")

print("\nDone.")
