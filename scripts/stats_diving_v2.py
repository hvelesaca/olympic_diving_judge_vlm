import scipy.io
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
import math

warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
plt.rcParams.update({
    'font.family'      : 'DejaVu Sans',
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'figure.dpi'       : 130,
    'savefig.dpi'      : 180,
    'savefig.bbox'     : 'tight',
})

# ══════════════════════════════════════════════════════════════════════════════
# PALETTE — maximally distinct colors (ColorBrewer + manual tuning)
# ══════════════════════════════════════════════════════════════════════════════
PALETTE = {
    # Qwen 7B — bold blues
    'Qwen 7B P1'        : '#003f88',   # deep navy
    'Qwen 7B P2'        : '#0077CC',   # vivid blue
    'Qwen 7B P3'       : '#00B4D8',   # cyan-blue

    # Qwen 3B — saturated purples/magentas
    'Qwen 3B P1'        : '#7B2FBE',   # deep violet
    'Qwen 3B P2'        : '#D81B60',   # hot magenta
    'Qwen 3B P3'       : '#F48FB1',   # soft rose

    # LLaVA — vivid greens
    'LLaVA P1'     : '#1B5E20',   # forest green
    'LLaVA P3'    : '#43A047',   # medium green

    # MiniCPM — bold orange
    'MiniCPM P1'        : '#E65100',   # deep orange

    # InternVL — reds/coral
    'InternVL P1'       : '#B71C1C',   # dark red
    'InternVL P2'       : '#F4511E',   # orange-red
    'InternVL P3'      : '#FF8A65',   # salmon

    # Qwen 32B — golds/amber
    'Qwen 32B P1'       : '#F9A825',   # amber
    'Qwen 32B P2'       : '#FDD835',   # yellow
    'Qwen 32B P3'      : '#FFEE58',   # pale yellow

    # Regressor — charcoal
    'Regressor Ensemble'          : '#37474F',
}

# ── Maximally distinct markers ─────────────────────────────────────────────────
MARKERS = {
    'Qwen 7B P1'        : 'o',    # circle
    'Qwen 7B P2'        : 's',    # square
    'Qwen 7B P3'       : '^',    # triangle up

    'Qwen 3B P1'        : 'D',    # diamond
    'Qwen 3B P2'        : 'v',    # triangle down
    'Qwen 3B P3'       : 'P',    # plus-filled

    'LLaVA P1'     : '*',    # star
    'LLaVA P3'    : 'h',    # hexagon

    'MiniCPM P1'        : 'X',    # x-filled

    'InternVL P1'       : 'p',    # pentagon
    'InternVL P2'       : '<',    # triangle left
    'InternVL P3'      : '>',    # triangle right

    'Qwen 32B P1'       : '8',    # octagon
    'Qwen 32B P2'       : 'd',    # thin diamond
    'Qwen 32B P3'      : 'H',    # hexagon2

    'Regressor Ensemble'          : '+',    # plus
}

# ══════════════════════════════════════════════════════════════════════════════
# ▶ CONFIGURABLE PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
TOP_N_FIG8   = 3     # Fig 8: number of top models to display
TOP_N_FIG10  = 5    # Fig 10: number of VLMs to display in radar chart

# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
mat = scipy.io.loadmat('D:/Henry/VLMs/diving/split_300_70/testing_idx.mat')
test_idx_py = mat['testing_idx'].flatten() - 1

print("test_idx_py:", test_idx_py)
print("test_idx_py len:", len(test_idx_py))

df = pd.read_excel('./evaluaciones_todos_videos_all_orin.xlsx', header=0)
for c in df.columns:
    df[c] = pd.to_numeric(df[c], errors='coerce') if df[c].dtype == object else df[c]

print(f"Total videos loaded: {len(df)}")
print("Head:\n", df.head())

# ══════════════════════════════════════════════════════════════════════════════
# 2. TRAIN / TEST SPLIT
# ══════════════════════════════════════════════════════════════════════════════
valid_test = test_idx_py[test_idx_py < len(df)]
train_idx  = np.array([i for i in range(len(df)) if i not in valid_test])

df_test  = df.iloc[valid_test].reset_index(drop=True)
df_train = df.iloc[train_idx].reset_index(drop=True)

print(f"Train: {len(df_train)} | Test: {len(df_test)}")

GT = df_test['overall_scores_GT'].values


# ── MODEL CONFIG ───────────────────────────────────────────────────────────────
MODEL_CONFIG = [
    ('Qwen 7B P1',  'qwen_7b_p6', 'qwen_7b_p6_inference_time_s',                    None),
    ('Qwen 7B P2',  'qwen_7b_p8', 'qwen_7b_p8_inference_time_s',                    'qwen'),
    ('Qwen 7B P3', 'qwen_7b_no_cond', 'qwen_7b_no_cond_inference_time_s',          None),
    ('Qwen 3B P1',  'qwen_3b_prompt_6', 'qwen_3b_prompt_6_inference_time_s',                                        None),
    ('Qwen 3B P2',  'qwen_3b_prompt_8', 'qwen_3b_prompt_8_inference_time_s',                                        None),
    ('Qwen 3B P3', 'qwen_3b_prompt_no_cond', 'qwen_3b_prompt_no_cond_inference_time_s', None),
    ('LLaVA P1',  'llava_p6_weighted',  'llava_inference_time_s',              'llava'),
    ('LLaVA P3', 'llava_no_cond',      'llava_no_cond_inference_time_s',      None),
    ('MiniCPM P1', 'minicpm_p6_weighted',    'minicpm_inference_time_s',            'minicpm'),
    ('InternVL P1',  'inter_vl3_p6',         'inter_vl3_p6_inference_time_s',       None),
    ('InternVL P2',  'inter_vl3_p8',         'inter_vl3_p8_inference_time_s',       None),
    ('InternVL P3', 'inter_vl3_no_cond',    'inter_vl3_no_cond_inference_time_s',  None),
    ('Qwen 32B P1',  'qwen_32B_p6',          'qwen_32B_p6_inference_time_s',        None),
    ('Qwen 32B P2',  'qwen_32B_p8',          'qwen_32B_p8_inference_time_s',        None),
    ('Qwen 32B P3', 'qwen_32B_no_cond',     'qwen_32B_no_cond_inference_time_s',   None),
    ('Regressor Ensemble',     'best_cal',             None,                                   None),
]
models_pred    = {cfg[0]: df_test[cfg[1]].values for cfg in MODEL_CONFIG if cfg[1] in df_test.columns}
colors         = {k: PALETTE.get(k, '#607D8B') for k in models_pred}
markers        = {k: MARKERS.get(k, 'o')       for k in models_pred}
time_col_map   = {cfg[0]: cfg[2] for cfg in MODEL_CONFIG if cfg[2] is not None}
sub_prefix_map = {cfg[0]: cfg[3] for cfg in MODEL_CONFIG if cfg[3] is not None}

N_MODELS = len(models_pred)

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def make_grid(n, max_cols=3):
    ncols = min(n, max_cols)
    nrows = math.ceil(n / ncols)
    return nrows, ncols

def make_subplots(n, max_cols=3, h_per_row=4.8, w_per_col=5.8, **kwargs):
    nrows, ncols = make_grid(n, max_cols)
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * w_per_col, nrows * h_per_row), **kwargs)
    axes = np.array(axes).flatten()
    for ax in axes[n:]:
        ax.set_visible(False)
    return fig, axes[:n]

def add_caption(fig, text, y=-0.01):
    fig.text(0.5, y, text, ha='center', va='top', fontsize=9.5,
             color='#455A64', style='italic',
             wrap=True, transform=fig.transFigure)

# ══════════════════════════════════════════════════════════════════════════════
# 3. METRICS
# ══════════════════════════════════════════════════════════════════════════════
def compute_metrics(gt, pred):
    mask = ~np.isnan(pred)
    g, p = gt[mask], pred[mask]
    return dict(
        Spearman = stats.spearmanr(g, p)[0],
        Pearson  = stats.pearsonr(g, p)[0],
        RMSE     = np.sqrt(mean_squared_error(g, p)),
        MAE      = mean_absolute_error(g, p),
        RelErr   = np.mean(np.abs(g - p) / (np.abs(g) + 1e-8)) * 100,
        R2       = r2_score(g, p),
        n        = mask.sum()
    )

results = {m: compute_metrics(GT, v) for m, v in models_pred.items()}

print("\n" + "=" * 74)
print(f"METRICS — Test Set (Split 300/70, N={len(df_test)})")
print("=" * 74)
print(f"{'Model':<18} {'Spearman ρ':>11} {'Pearson r':>10} {'RMSE':>8} {'MAE':>8} {'RelErr%':>9} {'R²':>8}")
print("-" * 74)
for m, r in results.items():
    print(f"{m:<18} {r['Spearman']:>11.4f} {r['Pearson']:>10.4f} "
          f"{r['RMSE']:>8.3f} {r['MAE']:>8.3f} {r['RelErr']:>9.2f} {r['R2']:>8.4f}")
print("=" * 74)

# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Scatter: Ground Truth vs Predicted Score
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = make_subplots(N_MODELS, max_cols=3, h_per_row=4.8, w_per_col=5.8)
lims = [GT.min() - 5, GT.max() + 5]

for ax, (name, pred) in zip(axes, models_pred.items()):
    mask = ~np.isnan(pred)
    g, p = GT[mask], pred[mask]
    col  = colors[name]
    mrk  = markers[name]

    ax.scatter(g, p, alpha=0.65, color=col, edgecolors='white',
               linewidths=0.5, s=55, marker=mrk, zorder=3)
    ax.plot(lims, lims, 'k--', lw=1.2, label='Ideal', zorder=2)

    m_fit, b = np.polyfit(g, p, 1)
    xs = np.linspace(lims[0], lims[1], 100)
    ax.plot(xs, m_fit * xs + b, color=col, lw=2,
            label=f'Fit (m={m_fit:.2f})', zorder=4)

    r = results[name]
    ax.set_title(name, fontsize=12, fontweight='bold', color=col, pad=6)
    ax.set_xlabel('Ground Truth Score (GT)', fontsize=10)
    ax.set_ylabel('Predicted Score', fontsize=10)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.legend(fontsize=8, loc='upper left')
    ax.text(0.97, 0.04,
            f"ρ={r['Spearman']:.3f}  r={r['Pearson']:.3f}\n"
            f"RMSE={r['RMSE']:.2f}  MAE={r['MAE']:.2f}",
            transform=ax.transAxes, va='bottom', ha='right', fontsize=8.5,
            bbox=dict(boxstyle='round,pad=0.35', fc='white', alpha=0.85, ec='#CFD8DC'))

fig.suptitle('FIG 1 — Scatter: Ground Truth vs Predicted Score\n'
             '(Test Set — Split 300/70)',
             fontsize=14, fontweight='bold', y=1.01)
add_caption(fig,
    "Each point is a video. The dashed diagonal represents perfect prediction. "
    "Points clustered along the diagonal indicate better model performance.")
plt.tight_layout()
plt.savefig('fig1_scatter.png')
plt.show()
print("fig1_scatter.png saved")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Spearman ρ Pareto Chart (vertical bars, no baselines)
#          • Sorted highest → lowest
#          • Models with negative Spearman excluded
# ══════════════════════════════════════════════════════════════════════════════
_names = list(results.keys())
_rhos  = [results[m]['Spearman'] for m in _names]
_cols  = [colors[m] for m in _names]

# Exclude negative Spearman
filtered = [(n, r, c) for n, r, c in zip(_names, _rhos, _cols) if r >= 0]
# Sort descending (Pareto)
filtered.sort(key=lambda x: x[1], reverse=True)
sorted_names, sorted_rhos, sorted_cols = zip(*filtered) if filtered else ([], [], [])

fig, ax = plt.subplots(figsize=(max(8, len(sorted_names) * 0.8 + 2), 6))
bars = ax.bar(sorted_names, sorted_rhos, color=sorted_cols,
              edgecolor='black', linewidth=0.7, width=0.65)

# Value labels on top of each bar
for bar, val in zip(bars, sorted_rhos):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f'{val:.3f}', va='bottom', ha='center',
            fontsize=11, fontweight='bold', color='#263238')

#ax.set_title("FIG 2 — Spearman ρ Pareto Chart: All VLMs\n"
#"(Test Set — Split 300/70, sorted highest → lowest)",
#fontsize=13, fontweight='bold')
ax.set_ylim(0, 0.70)
ax.tick_params(axis='x', rotation=90, labelsize=11)
ax.grid(axis='y', alpha=0.35)

#add_caption(fig,
#    "A higher ρ means the model better preserves the relative ranking of divers. "
#    "Models with negative Spearman ρ are excluded. Bars sorted from best to worst.")
plt.tight_layout()
plt.savefig('fig2_spearman_comparison.png')
plt.show()
print("fig2_spearman_comparison.png saved")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Residual Distribution
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = make_subplots(N_MODELS, max_cols=3, h_per_row=4.5, w_per_col=5.6)

for ax, (name, pred) in zip(axes, models_pred.items()):
    mask      = ~np.isnan(pred)
    residuals = GT[mask] - pred[mask]
    col       = colors[name]

    sns.histplot(residuals, bins=20, color=col, alpha=0.80,
                 edgecolor='white', ax=ax, kde=True, line_kws={'lw': 2})
    ax.axvline(0,                color='black', lw=1.5, linestyle='--', label='Zero')
    ax.axvline(residuals.mean(), color='red',   lw=1.5, linestyle='-',
               label=f'Mean={residuals.mean():.2f}')
    ax.set_title(name, fontsize=12, fontweight='bold', color=col, pad=6)
    ax.set_xlabel('Residual (GT − Prediction)', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.legend(fontsize=8)
    r = results[name]
    ax.text(0.97, 0.96, f"MAE={r['MAE']:.2f}\nRMSE={r['RMSE']:.2f}",
            transform=ax.transAxes, va='top', ha='right', fontsize=8.5,
            bbox=dict(boxstyle='round,pad=0.35', fc='white', alpha=0.85, ec='#CFD8DC'))

fig.suptitle('FIG 3 — Residual Distribution by Model\n'
             '(Test Set — Split 300/70)',
             fontsize=14, fontweight='bold', y=1.01)
add_caption(fig,
    "A histogram centered at 0 and symmetric indicates no systematic bias. "
    "The KDE curve smooths the distribution. The red line marks the mean error.")
plt.tight_layout()
plt.savefig('fig3_residuals.png')
plt.show()
print("fig3_residuals.png saved")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Score Distribution: GT vs Predictions
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = make_subplots(N_MODELS, max_cols=3, h_per_row=4.5, w_per_col=5.6)
bins = np.linspace(30, 110, 25)

for ax, (name, pred) in zip(axes, models_pred.items()):
    mask = ~np.isnan(pred)
    col  = colors[name]
    sns.histplot(GT[mask],   bins=bins, color='#90A4AE', alpha=0.55,
                 label='GT (Real)', ax=ax, edgecolor='white', kde=False)
    sns.histplot(pred[mask], bins=bins, color=col, alpha=0.70,
                 label=name,        ax=ax, edgecolor='white', kde=True,
                 line_kws={'lw': 2})
    ax.set_title(name, fontsize=12, fontweight='bold', color=col, pad=6)
    ax.set_xlabel('Score', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.legend(fontsize=8)

fig.suptitle('FIG 4 — Score Distribution: GT vs Prediction\n'
             '(Test Set — Split 300/70)',
             fontsize=14, fontweight='bold', y=1.01)
add_caption(fig,
    "The grey histogram represents the real jury scores. "
    "The coloured histogram shows model predictions. "
    "High overlap indicates the model closely replicates the real score distribution.")
plt.tight_layout()
plt.savefig('fig4_score_distribution.png')
plt.show()
print("fig4_score_distribution.png saved")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 5 — Spearman ρ by Difficulty Degree (DD)
# ══════════════════════════════════════════════════════════════════════════════
diff      = df_test['difficulty_level_GT'].values
diff_vals = np.sort(df_test['difficulty_level_GT'].dropna().unique())

fig_w5 = max(12, N_MODELS * 1.8 + len(diff_vals) * 0.7)
fig, ax = plt.subplots(figsize=(fig_w5, 6))
x     = np.arange(len(diff_vals))
width = min(0.8 / N_MODELS, 0.22)

for i, (name, pred) in enumerate(models_pred.items()):
    rhos = []
    for dv in diff_vals:
        mask = (diff == dv) & ~np.isnan(pred)
        rhos.append(stats.spearmanr(GT[mask], pred[mask])[0]
                    if mask.sum() >= 5 else np.nan)
    offset = (i - N_MODELS / 2 + 0.5) * width
    bars = ax.bar(x + offset, rhos, width, label=name,
                  color=colors[name], edgecolor='white', linewidth=0.5,
                  alpha=0.88)
    for bar, val in zip(bars, rhos):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01, f'{val:.2f}',
                    ha='center', va='bottom', fontsize=6.5, rotation=90)

ax.set_xticks(x)
ax.set_xticklabels([f'DD={d:.1f}' for d in diff_vals], fontsize=10)
ax.set_ylabel("Spearman ρ", fontsize=12)
ax.set_title("FIG 5 — Spearman ρ by Difficulty Degree (DD)\n"
             "(Test Set — Split 300/70)",
             fontsize=13, fontweight='bold')
ax.legend(fontsize=8.5, ncol=3, loc='lower right')
ax.axhline(0, color='black', lw=0.8)
ax.set_ylim(-0.55, 1.15)

add_caption(fig,
    "Difficulty Degree (DD) is the official dive multiplier. "
    "A consistent ρ across all DD values indicates the model generalises well "
    "regardless of the technical complexity of the dive.")
plt.tight_layout()
plt.savefig('fig5_spearman_by_difficulty.png')
plt.show()
print("fig5_spearman_by_difficulty.png saved")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 6 — Mean Sub-scores by Dive Phase
# ══════════════════════════════════════════════════════════════════════════════
phases       = ['starting', 'approach', 'take_off', 'flight', 'entry']
phase_labels = ['Starting\nPosition', 'Approach', 'Take-off', 'Flight', 'Entry']
sub_models   = [(name, sub_prefix_map[name], colors[name])
                for name in models_pred if name in sub_prefix_map]
n_sub = len(sub_models)

if n_sub > 0:
    fig, axes = plt.subplots(1, n_sub,
                              figsize=(n_sub * 6, 5.5), sharey=True)
    axes = np.array(axes).flatten()

    for ax, (name, prefix, col) in zip(axes, sub_models):
        means, stds = [], []
        for ph in phases:
            col_name = f'{prefix}_sub_score_{ph}'
            if col_name in df_test.columns:
                vals = pd.to_numeric(df_test[col_name], errors='coerce').dropna()
                means.append(vals.mean()); stds.append(vals.std())
            else:
                means.append(np.nan); stds.append(np.nan)
        x = np.arange(len(phases))
        bars = ax.bar(x, means, yerr=stds, color=col, alpha=0.82,
                      edgecolor='white', linewidth=0.7, capsize=5)
        ax.set_xticks(x)
        ax.set_xticklabels(phase_labels, fontsize=10)
        ax.set_title(name, fontsize=13, fontweight='bold', color=col, pad=6)
        ax.set_ylabel('Mean Sub-score', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        for bar, m_val in zip(bars, means):
            if not np.isnan(m_val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.08,
                        f'{m_val:.2f}', ha='center', va='bottom', fontsize=9)

    fig.suptitle('FIG 6 — Mean Sub-scores by Dive Phase\n'
                 '(Test Set — Split 300/70)',
                 fontsize=14, fontweight='bold')
    add_caption(fig,
        "Error bars = ±1 standard deviation. "
        "Shows which dive phase each VLM evaluates best or worst. "
        "Only models with phase-level sub-scores are included.")
    plt.tight_layout()
    plt.savefig('fig6_subscores_by_phase.png')
    plt.show()
    print("fig6_subscores_by_phase.png saved")
else:
    print("FIG 6 skipped: no model has a sub_score_prefix defined.")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 7 — MAE and Spearman by GT Score Quartile
# ══════════════════════════════════════════════════════════════════════════════
quartiles = pd.qcut(GT, q=4, labels=['Q1\n(Low)', 'Q2', 'Q3', 'Q4\n(High)'])
fig, axes = plt.subplots(1, 2, figsize=(max(14, N_MODELS * 1.8), 5.5))
x     = np.arange(4)
width = min(0.8 / N_MODELS, 0.2)

for i, (name, pred) in enumerate(models_pred.items()):
    maes, rhos = [], []
    for q in ['Q1\n(Low)', 'Q2', 'Q3', 'Q4\n(High)']:
        mask = (quartiles == q) & ~np.isnan(pred)
        if mask.sum() >= 3:
            maes.append(mean_absolute_error(GT[mask], pred[mask]))
            rhos.append(stats.spearmanr(GT[mask], pred[mask])[0]
                        if mask.sum() >= 5 else np.nan)
        else:
            maes.append(np.nan); rhos.append(np.nan)
    offset = (i - N_MODELS / 2 + 0.5) * width
    axes[0].bar(x + offset, maes, width, label=name,
                color=colors[name], edgecolor='white', lw=0.5, alpha=0.88)
    axes[1].bar(x + offset, rhos, width, label=name,
                color=colors[name], edgecolor='white', lw=0.5, alpha=0.88)

for ax, ylabel, title in zip(axes,
        ['MAE', "Spearman ρ"],
        ['MAE by GT Score Quartile', 'Spearman ρ by GT Score Quartile']):
    ax.set_xticks(x)
    ax.set_xticklabels(['Q1\n(Low)', 'Q2', 'Q3', 'Q4\n(High)'], fontsize=10)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=7.5, ncol=2)
    ax.grid(axis='y', alpha=0.3)
axes[1].axhline(0, color='black', lw=0.8)

fig.suptitle('FIG 7 — Performance by Score Range\n'
             '(Test Set — Split 300/70)',
             fontsize=13, fontweight='bold')
add_caption(fig,
    "Q1=lowest scores, Q4=highest. "
    "Rising MAE at Q4 indicates the model struggles with high-scoring dives. "
    "Low ρ in a quartile signals loss of ranking ability in that range.")
plt.tight_layout()
plt.savefig('fig7_performance_by_quartile.png')
plt.show()
print("fig7_performance_by_quartile.png saved")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 8 — GT vs Predicted Score per Video
#          Shows only the top TOP_N_FIG8 models by Spearman ρ (configurable)
# ══════════════════════════════════════════════════════════════════════════════
# Select top N models by Spearman
top_models_fig8 = sorted(
    models_pred.keys(),
    key=lambda m: results[m]['Spearman'],
    reverse=True
)[:TOP_N_FIG8]

sort_order = np.argsort(valid_test)
fig, ax = plt.subplots(figsize=(16, 5.5))
ax.plot(GT[sort_order], 'o-', color='#455A64', lw=1.8, ms=4,
        label='Ground Truth (GT)', alpha=0.95, zorder=5)

for name in top_models_fig8:
    pred = models_pred[name]
    ax.plot(pred[sort_order], linestyle='--', color=colors[name],
            marker=markers[name], lw=1.2, ms=3,
            label=f'{name}  (ρ={results[name]["Spearman"]:.3f})',
            alpha=0.82, markevery=4)

ax.set_xlabel('Test video (sorted by index)', fontsize=11)
ax.set_ylabel('Score', fontsize=11)
#ax.set_title(f'FIG 8 — GT vs Predicted Score per Test Video\n'
#             f'(Top {TOP_N_FIG8} models by Spearman ρ — Split 300/70)',
#             fontsize=13, fontweight='bold')
ax.legend(fontsize=11, ncol=2, loc='upper right',
          framealpha=0.92, edgecolor='#CFD8DC')
ax.grid(True, alpha=0.3)

#add_caption(fig,
#    f"Only the top {TOP_N_FIG8} models ranked by Spearman ρ are shown (configurable via TOP_N_FIG8). "
#    "A curve closely following the grey GT line indicates high pointwise fidelity.")
plt.tight_layout()
plt.savefig('fig8_gt_vs_pred_per_video.png')
plt.show()
print("fig8_gt_vs_pred_per_video.png saved")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 9 — Inference Time: Mean ± Std bar chart only (right panel)
# ══════════════════════════════════════════════════════════════════════════════
time_data, time_labels, time_colors = [], [], []
for name, col_t in time_col_map.items():
    if col_t in df_test.columns:
        vals = pd.to_numeric(df_test[col_t], errors='coerce').dropna().values
        time_data.append(vals)
        time_labels.append(name)
        time_colors.append(colors[name])

if time_data:
    fig, ax = plt.subplots(figsize=(max(10, len(time_labels) * 1.1 + 2), 6))

    means = [np.mean(d) for d in time_data]
    stds  = [np.std(d)  for d in time_data]

    # Sort by mean inference time (ascending = fastest first)
    order_t = np.argsort(means)
    t_labels_s = [time_labels[i] for i in order_t]
    t_means_s  = [means[i]        for i in order_t]
    t_stds_s   = [stds[i]         for i in order_t]
    t_colors_s = [time_colors[i]  for i in order_t]

    bars = ax.bar(t_labels_s, t_means_s, yerr=t_stds_s,
                  color=t_colors_s, edgecolor='white', linewidth=0.7,
                  capsize=7, alpha=0.88)
    for bar, m_val, s in zip(bars, t_means_s, t_stds_s):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + s + 0.4,
                f'{m_val:.1f}s', ha='center', va='bottom',
                fontsize=11, fontweight='bold')
    ax.set_ylabel('Mean Inference Time (s)', fontsize=12)
    #ax.set_title('FIG 9 — Mean ± Std. Dev. of Inference Time\n'
    #             '(Test Set — Split 300/70, sorted fastest → slowest)',
    #             fontsize=13, fontweight='bold')
    ax.tick_params(axis='x', rotation=90, labelsize=9)
    ax.grid(axis='y', alpha=0.3)

    #add_caption(fig,
    #    "Bars show mean inference time ± 1 standard deviation per model. "
    #    "Lower time = higher viability for real-time applications. "
    #    "Sorted from fastest to slowest.")
    plt.tight_layout()
    plt.savefig('fig9_inference_time.png')
    plt.show()
    print("fig9_inference_time.png saved")
else:
    print("FIG 9 skipped: no inference time columns available.")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 10 — Radar Chart (normalised metrics)
#           Shows top TOP_N_FIG10 models by Spearman ρ (configurable)
# ══════════════════════════════════════════════════════════════════════════════
# ── Data-driven normalization bounds (avoids arbitrary fixed constants) ────────
_max_rmse   = max(r['RMSE']   for r in results.values())
_max_mae    = max(r['MAE']    for r in results.values())
_max_relerr = max(r['RelErr'] for r in results.values())
# Use a small epsilon to avoid division by zero if all errors are identical
_max_rmse   = _max_rmse   if _max_rmse   > 0 else 1.0
_max_mae    = _max_mae    if _max_mae    > 0 else 1.0
_max_relerr = _max_relerr if _max_relerr > 0 else 1.0

metric_labels = [
    'Spearman ρ',
    'Pearson r',
    f'1−RMSE', #/{_max_rmse:.1f}
    f'1−MAE', #/{_max_mae:.1f}
    f'1−RelErr', #/{_max_relerr:.1f}
    'R²(norm)',
]

def normalize_metrics(r):
    return [
        max(r['Spearman'], 0),
        max(r['Pearson'],  0),
        max(1 - r['RMSE']   / _max_rmse,   0),
        max(1 - r['MAE']    / _max_mae,    0),
        max(1 - r['RelErr'] / _max_relerr, 0),
        max((r['R2'] + 1) / 2, 0),
    ]

# Select top N models by Spearman for radar
top_models_fig10 = sorted(
    results.keys(),
    key=lambda m: results[m]['Spearman'],
    reverse=True
)[:TOP_N_FIG10]

N_met  = len(metric_labels)
angles = np.linspace(0, 2 * np.pi, N_met, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
for name in top_models_fig10:
    r    = results[name]
    vals = normalize_metrics(r) + [normalize_metrics(r)[0]]
    ax.plot(angles, vals, 'o-', lw=2, label=name,
            color=colors[name], marker=markers[name], ms=6)
    ax.fill(angles, vals, alpha=0.08, color=colors[name])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metric_labels, fontsize=11)
ax.set_ylim(0, 1)
#ax.set_title(f'FIG 10 — Normalised Metrics Radar Chart\n'
#             f'(Top {TOP_N_FIG10} models by Spearman ρ — Test Set — Split 300/70)',
#             fontsize=13, fontweight='bold', pad=22)
ax.legend(loc='upper right', bbox_to_anchor=(1.45, 1.15), fontsize=12)
ax.grid(True, alpha=0.4)

#fig.text(0.5, 0.01,
#    f"Top {TOP_N_FIG10} models shown (configurable via TOP_N_FIG10). "
#    "Each axis is normalised to [0,1]. A larger polygon = better overall profile.",
#    ha='center', fontsize=9, color='#455A64', style='italic')
plt.tight_layout()
plt.savefig('fig10_radar.png')
plt.show()
print("fig10_radar.png saved")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 11 — Metrics Summary Table
# ══════════════════════════════════════════════════════════════════════════════
baselines_table = {
    'C3D-SVR [P1]' : dict(Spearman=0.74, Pearson=np.nan, RMSE=np.nan, MAE=np.nan, RelErr=np.nan, R2=np.nan),
    'LSTM [P1]'    : dict(Spearman=0.62, Pearson=np.nan, RMSE=np.nan, MAE=np.nan, RelErr=np.nan, R2=np.nan),
    'C3D-LSTM [P2]': dict(Spearman=0.82, Pearson=np.nan, RMSE=np.nan, MAE=np.nan, RelErr=np.nan, R2=np.nan),
    'USDL [P2]'    : dict(Spearman=0.87, Pearson=np.nan, RMSE=np.nan, MAE=np.nan, RelErr=np.nan, R2=np.nan),
}
all_results = {**baselines_table, **results}

fig, ax = plt.subplots(figsize=(16, max(7, len(all_results) * 0.55 + 2.5)))
ax.axis('off')
col_labels = ['Method', 'Spearman ρ↑', 'Pearson r↑', 'RMSE↓', 'MAE↓', 'RelErr%↓', 'R²↑']
rows = []
for name, r in all_results.items():
    rows.append([
        name,
        f"{r['Spearman']:.4f}" if not np.isnan(r['Spearman']) else '—',
        f"{r['Pearson']:.4f}"  if not np.isnan(r['Pearson'])  else '—',
        f"{r['RMSE']:.3f}"     if not np.isnan(r['RMSE'])     else '—',
        f"{r['MAE']:.3f}"      if not np.isnan(r['MAE'])       else '—',
        f"{r['RelErr']:.2f}"   if not np.isnan(r['RelErr'])    else '—',
        f"{r['R2']:.4f}"       if not np.isnan(r['R2'])        else '—',
    ])

table = ax.table(cellText=rows, colLabels=col_labels,
                  loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10.5)
table.scale(1.15, 2.1)

for j in range(len(col_labels)):
    cell = table[0, j]
    cell.set_facecolor('#263238')
    cell.set_text_props(color='white', fontweight='bold')

for i in range(1, len(rows) + 1):
    is_baseline = i <= len(baselines_table)
    row_color = '#ECEFF1' if is_baseline else '#E8F5E9'
    if not is_baseline:
        try:
            rho_val  = float(rows[i - 1][1])
            best_rho = max(results[m]['Spearman'] for m in results)
            if abs(rho_val - best_rho) < 1e-4:
                row_color = '#B2DFDB'
        except Exception:
            pass
    for j in range(len(col_labels)):
        table[i, j].set_facecolor(row_color)

ax.set_title('FIG 11 — Performance Metrics Table\n'
             'UNLV-Dive Split 300/70  |  Grey: Literature Baselines  |  '
             'Green: This Work  |  Teal: Best VLM',
             fontsize=12, fontweight='bold', pad=18)
plt.tight_layout()
plt.savefig('fig11_metrics_table.png')
plt.show()
print("fig11_metrics_table.png saved")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 12 — Prediction Correlation Heatmap
# ══════════════════════════════════════════════════════════════════════════════
pred_df     = pd.DataFrame({name: pred for name, pred in models_pred.items()})
corr_matrix = pred_df.corr(method='spearman')

fig, ax = plt.subplots(figsize=(max(10, N_MODELS * 0.9), max(8, N_MODELS * 0.75)))
mask_upper = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
            vmin=-1, vmax=1, ax=ax, linewidths=0.4, linecolor='white',
            annot_kws={'size': 8.5}, mask=mask_upper,
            cbar_kws={'label': 'Spearman ρ between predictions'})
ax.set_title('FIG 12 — Prediction Correlation Heatmap (VLMs)\n'
             '(Spearman ρ — Test Set)',
             fontsize=13, fontweight='bold', pad=12)
ax.tick_params(axis='x', rotation=90, labelsize=9)
ax.tick_params(axis='y', rotation=0,  labelsize=9)

add_caption(fig,
    "Green = high correlation (similar models). Red = low correlation (diverse models). "
    "Pairs with ρ < 0.85 are good candidates for ensembling due to prediction diversity.")
plt.tight_layout()
plt.savefig('fig12_correlation_heatmap.png')
plt.show()
print("fig12_correlation_heatmap.png saved")

print("\n✅ All figures generated successfully.")
print("\n=== COLOR LEGEND BY MODEL FAMILY ===")
for fam, col_range in [
    ("Qwen 7B   (P6/P8/P11)", "Navy / Vivid Blue / Cyan-Blue"),
    ("Qwen 3B   (P6/P8/P11)", "Deep Violet / Hot Magenta / Soft Rose"),
    ("LLaVA Next (P6/P11)",   "Forest Green / Medium Green"),
    ("MiniCPM    (P6)",       "Deep Orange"),
    ("InternVL   (P6/P8/P11)","Dark Red / Orange-Red / Salmon"),
    ("Qwen 32B   (P6/P8/P11)","Amber / Yellow / Pale Yellow"),
    ("Regressor",             "Charcoal"),
]:
    print(f"  {fam:<30} → {col_range}")