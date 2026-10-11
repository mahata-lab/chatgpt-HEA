"""API-only Figure 6 and per-case box-plot alternative."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / 'manuscript_figures' / 'reference_images'
DATA = ROOT.parent / 'reference_outputs'
tiers = ['Simple (Binary)', 'Complex (Ternary)', 'Very Complex (Quinary)']
short = ['Binary', 'Ternary', 'Quinary']
colors = ['#1b7837', '#e08214', '#b2182b']
case = pd.read_csv(DATA / 'reproducibility_by_system.csv')
metrics = pd.read_csv(DATA / 'metrics_reproducibility.csv').set_index('tier').loc[tiers]
assert len(case) == 24 and all(sum(case.tier == t) == 8 for t in tiers)
for t in tiers:
    assert np.isclose(case.loc[case.tier == t, 'SD_dH_kJmol'].mean(), metrics.loc[t, 'mean_SD_dH_kJmol'])
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                    'axes.edgecolor': '#333333', 'axes.linewidth': 1,
                    'savefig.dpi': 300})
x = np.arange(3)
fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5))
ax = axes[0]
vals = metrics.mean_SD_dH_kJmol.to_numpy()
ax.bar(x, vals, .6, color=colors, edgecolor='white')
for i, v in enumerate(vals): ax.text(i, v + .05, f'{v:.2f}', ha='center', fontweight='bold')
ax.set(xticks=x, xticklabels=short, ylim=(0, 2.5),
       ylabel=r'Mean within-case SD of $\Delta H_{mix}$ (kJ mol$^{-1}$)')
ax.set_title('(a) Numerical repeatability\nFive responses per named alloy case', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=.22)
ax = axes[1]
vals = metrics.categorical_agreement_pct.to_numpy()
ax.plot(x, vals, 'o-', color='#1a9850', lw=2, markersize=8)
for i, v in enumerate(vals): ax.text(i, v + 2, f'{v:.0f}%', ha='center', fontweight='bold')
ax.set(xticks=x, xticklabels=short, ylim=(0, 110), xlim=(-.3, 2.3),
       ylabel='Mean majority phase agreement (%)')
ax.set_title('(b) Phase repeatability\nFive responses per named alloy case', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=.22)
fig.tight_layout()
fig.savefig(OUT / 'Fig.6_repeatability.png', bbox_inches='tight')
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8))
for ax, key, title, ylabel in zip(axes,
    ['SD_dH_kJmol', 'SD_delta_pctpoint'],
    ['(a) Variation in mixing-enthalpy estimates', '(b) Variation in atomic-size mismatch estimates'],
    [r'Within-case SD of $\Delta H_{mix}$ (kJ mol$^{-1}$)', r'Within-case SD of $\delta$ (percentage points)']):
    groups = [case.loc[case.tier == t, key].to_numpy() for t in tiers]
    bp = ax.boxplot(groups, positions=x, widths=.45, patch_artist=True, showfliers=False,
                    medianprops={'color': '#111111', 'linewidth': 2})
    for patch, c in zip(bp['boxes'], colors): patch.set(facecolor=c, alpha=.3)
    for i, (t, values) in enumerate(zip(tiers, groups)):
        offsets = np.linspace(-.13, .13, len(values))
        ax.scatter(i + offsets, values, s=38, color=colors[i], edgecolor='white', linewidth=.5, zorder=3)
        if key == 'SD_dH_kJmol':
            g = case[case.tier == t].reset_index(drop=True)
            for j, r in g.iterrows():
                if r[key] > 10:
                    ax.annotate(f"{r.system}: {r[key]:.2f}", (i + offsets[j], r[key]),
                                xytext=(16, -2), textcoords='offset points', fontsize=9)
    ax.set(xticks=x, xticklabels=short, ylabel=ylabel)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=.22)
axes[0].set_yscale('symlog', linthresh=.25)
ticks = [0, .25, .5, 1, 2, 4, 8, 16]
axes[0].yaxis.set_major_locator(FixedLocator(ticks))
axes[0].yaxis.set_major_formatter(FixedFormatter(['0', '0.25', '0.5', '1', '2', '4', '8', '16']))
axes[0].set_ylim(-.04, 25)
axes[1].set_ylim(bottom=-.08)
fig.text(.5, .015, 'Dots: eight named cases per tier; boxes: quartiles and median. Panel (a): nonlinear scale above 0.25.',
         ha='center', fontsize=9)
fig.tight_layout(rect=[0, .055, 1, 1])
fig.savefig(OUT / 'Fig.6_repeatability_boxplots.png', bbox_inches='tight')
plt.close(fig)
print('Saved Fig.6_repeatability.png and Fig.6_repeatability_boxplots.png')
