import itertools
import pandas as pd
import numpy as np
import warnings
import time
import os
from scipy.io import loadmat
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import RobustScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')
t0 = time.time()

# ============================================================
#  DETECCIÓN DE GPU (cuML de RAPIDS)
#  Si cuML está disponible se usan sus versiones GPU;
#  si no, se cae automáticamente a sklearn.
# ============================================================
GPU_AVAILABLE = False
try:
    import cuml
    from cuml.ensemble import RandomForestRegressor as cuRF
    from cuml.linear_model import Ridge as cuRidge
    from cuml.svm import SVR as cuSVR
    from cuml.preprocessing import RobustScaler as cuRobustScaler
    GPU_AVAILABLE = True
    print("✅  cuML detectado → modelos correrán en GPU")
except ImportError:
    print("⚠️   cuML no encontrado → usando sklearn en CPU (pip install cuml para GPU)")

# ============================================================
#  CONFIG
# ============================================================
VLM_SOURCES = [
    {"name": "Qwen25VL_7B_p6",      "csv": "./Qwen25VL_7B/results_prompt6/reasoning.csv"},
    {"name": "Qwen25VL_7B_p8",      "csv": "./Qwen25VL_7B/results_prompt8/reasoning.csv"},
    {"name": "llava_p6",             "csv": "./LlaVA Next 7B/results_prompt6/reasoning.csv"},
    #{"name": "llava_p11",             "csv": "./LlaVA Next 7B/results_no_cond/reasoning.csv"},
    #{"name": "Qwen25VL_7B_p6",      "csv": "./Qwen25VL_3B/results_prompt6/reasoning.csv"},
    {"name": "Qwen25VL_3B_p8",      "csv": "./Qwen25VL_3B/results_prompt8/reasoning.csv"},
    {"name": "Qwen25VL_3B_no_cond", "csv": "./Qwen25VL_3B/results_no_cond/reasoning.csv"},
    {"name": "Qwen25VL_32B_no_cond", "csv": "./Qwen25VL_32B/results_no_cond/reasoning.csv"},
    {"name": "Qwen25VL_32B_p6",      "csv": "./Qwen25VL_32B/results_prompt6/reasoning.csv"},
    {"name": "Qwen25VL_32B_p8",      "csv": "./Qwen25VL_32B/results_prompt8/reasoning.csv"},
    {"name": "InternVL3_8B_no_cond", "csv": "./InternVL3_8B/results_no_cond/reasoning.csv"},
    {"name": "InternVL3_8B_p6",      "csv": "./InternVL3_8B/results_prompt6/reasoning.csv"},
    {"name": "InternVL3_8B_p8",      "csv": "./InternVL3_8B/results_prompt8/reasoning.csv"},
]

# Tamaños de combinaciones a evaluar
# None  → todos los tamaños posibles (1, 2, ..., N)
# [1]   → solo individuales
# [1,2] → individuales + pares
COMBO_SIZES = [5]  # ← CAMBIA AQUÍ

GROUND_TRUTH_EXCEL = "./evaluaciones_todos_videos_all.xlsx"
TRAIN_IDX_MAT      = "./diving/split_300_70/training_idx.mat"
TEST_IDX_MAT       = "./diving/split_300_70/testing_idx.mat"
SCORE_FEATS        = ["difficulty_level_GT"]
OUTPUT_CSV         = "predicciones_combinaciones.csv"

SUB_SCORE_COLS = [
    "sub_score_starting", "sub_score_approach",
    "sub_score_take_off", "sub_score_flight", "sub_score_entry",
]
QOE_TEXT_COLS = [
    "QoE_Starting_Position", "QoE_Approach",
    "QoE_Take_off", "QoE_Flight", "QoE_Entry",
]

# ── NOTA CONFIG ─────────────────────────────────────────────────
APPLY_NOTA           = True
NOTA_DIFFICULTY_COL  = "difficulty_level_GT"

def compute_nota(pred, difficulty):
    return pred * 3 * difficulty

# ── Paralelismo ──────────────────────────────────────────────────────────────
# N_JOBS_EXPERIMENTS: cuántos experimentos correr en paralelo.
#   -1 → todos los núcleos disponibles
#    1 → secuencial (útil para debug)
#    k → k procesos
N_JOBS_EXPERIMENTS = -1   # ← paralelismo entre experimentos

# Cuando se usa CPU-sklearn los RF internos también pueden usar n_jobs,
# pero conviene no sobrecomprometerse si ya hay paralelismo externo.
RF_NJOBS_INNER = 2  # ← núcleos por RF cuando hay paralelismo externo

assert len(VLM_SOURCES) >= 1, "Define al menos 1 VLM."

# ── Resolver tamaños ──────────────────────────────────────────────────────────
N      = len(VLM_SOURCES)
sizes  = COMBO_SIZES if COMBO_SIZES is not None else list(range(1, N + 1))
sizes  = [s for s in sizes if 1 <= s <= N]

experiments = []
for s in sizes:
    if s == 1:
        experiments += [(vlm,) for vlm in VLM_SOURCES]
    else:
        experiments += list(itertools.permutations(VLM_SOURCES, s))

total_experiments = len(experiments)
print(f"VLMs definidos ({N}): {[v['name'] for v in VLM_SOURCES]}")
print(f"Tamaños evaluados: {sizes}")
print(f"Total de experimentos: {total_experiments}")
for exp in experiments:
    print(f"  → {' + '.join(v['name'] for v in exp)}")
print()

# ── Cargar GT y splits una sola vez ──────────────────────────────────────────
print("[INIT] Cargando GT y splits...", flush=True)
gt_df        = pd.read_excel(GROUND_TRUTH_EXCEL)
test_idx     = loadmat(TEST_IDX_MAT)['testing_idx'].flatten()
test_idx_py  = test_idx - 1
train_idx_py = np.array([i for i in range(len(gt_df)) if i not in set(test_idx_py)])

# ── Pre-cargar todos los CSVs de VLMs ────────────────────────────────────────
print("[INIT] Pre-cargando CSVs de VLMs...", flush=True)
vlm_cache = {}
for vlm in VLM_SOURCES:
    name = vlm["name"]
    vdf  = pd.read_csv(vlm["csv"], sep=";")
    rename = {col: f"{name}_{col}" for col in SUB_SCORE_COLS + QOE_TEXT_COLS if col in vdf.columns}
    vdf  = vdf.rename(columns=rename)
    for col in [f"{name}_{c}" for c in SUB_SCORE_COLS if f"{name}_{c}" in vdf.columns]:
        vdf[col] = pd.to_numeric(
            vdf[col].astype(str).str.replace(',', '.', regex=False).str.strip(),
            errors='coerce'
        )
    vlm_cache[name] = vdf[["video"] + list(rename.values())]
    print(f"    VLM '{name}': {len(vdf)} filas  ({vlm['csv']})")

# ── Fábrica de modelos (GPU vs CPU) ──────────────────────────────────────────
def make_rf(n_jobs_inner=RF_NJOBS_INNER, **extra):
    base = dict(n_estimators=1000, max_depth=10, max_features=0.6,
                random_state=42, **extra)
    if GPU_AVAILABLE:
        # cuML RF no acepta max_features float → usar int approx
        n_feat = base.pop("max_features", None)
        return cuRF(**base)           # cuML distribuye en GPU automáticamente
    return RandomForestRegressor(**base, n_jobs=n_jobs_inner)

def make_gbr(**extra):
    base = dict(n_estimators=1000, max_depth=3, learning_rate=0.05,
                random_state=42, **extra)
    # cuML no tiene GBR; se usa sklearn siempre (también funciona en CPU)
    return GradientBoostingRegressor(**base)

def make_ridge(alpha=50.0):
    if GPU_AVAILABLE:
        return cuRidge(alpha=alpha)
    return Ridge(alpha=alpha)

def make_svr_pipeline():
    if GPU_AVAILABLE:
        # cuML SVR + cuML scaler
        from sklearn.pipeline import Pipeline as SKPipeline
        return SKPipeline([
            ("s", cuRobustScaler()),
            ("m", cuSVR(C=1.0)),
        ])
    return Pipeline([("s", RobustScaler()), ("m", SVR(C=1.0))])

# ── CV paralelo para búsqueda de hiperparámetro ──────────────────────────────
def _cv_fold(tr, va, SC_tr, y_tr, msl):
    """Evalúa un fold del KFold para un valor de min_samples_leaf."""
    m = make_rf(min_samples_leaf=msl)
    m.fit(SC_tr[tr], y_tr[tr])
    pred = m.predict(SC_tr[va])
    # cuML devuelve array de cupy; convertir a numpy
    if GPU_AVAILABLE:
        pred = np.asarray(pred)
    return spearmanr(y_tr[va], pred)[0]

def best_min_samples_leaf(SC_tr, y_tr, n_splits=5, n_jobs_cv=4):
    """
    Busca el mejor min_samples_leaf en {1..5} usando KFold.
    Los folds se evalúan en paralelo con joblib.
    """
    kf     = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = list(kf.split(SC_tr))

    best_cv, best_p = -1, 1
    for msl in [1, 2, 3, 4, 5]:
        scores = Parallel(n_jobs=n_jobs_cv, prefer="threads")(
            delayed(_cv_fold)(tr, va, SC_tr, y_tr, msl)
            for tr, va in splits
        )
        cv = float(np.mean(scores))
        if cv > best_cv:
            best_cv, best_p = cv, msl
    return best_p

# ── Pipeline de un experimento ────────────────────────────────────────────────
def run_experiment(vlm_subset, exp_idx, total):
    pair_label = " + ".join(v["name"] for v in vlm_subset)
    group_size = len(vlm_subset)

    df = gt_df.copy()
    for vlm in vlm_subset:
        df = df.merge(vlm_cache[vlm["name"]], on="video", how="left")

    train_df = df.iloc[train_idx_py].reset_index(drop=True)
    test_df  = df.iloc[test_idx_py].reset_index(drop=True)

    y_tr = train_df['GT_score'].values
    y_te = test_df['GT_score'].values

    all_sub_score_cols = [f"{v['name']}_{c}" for v in vlm_subset for c in SUB_SCORE_COLS]
    all_qoe_text_cols  = [f"{v['name']}_{c}" for v in vlm_subset for c in QOE_TEXT_COLS]

    txt_tr = train_df[all_qoe_text_cols].fillna('').astype(str).agg(' '.join, axis=1).values
    txt_te = test_df[all_qoe_text_cols].fillna('').astype(str).agg(' '.join, axis=1).values

    num_cols = all_sub_score_cols + SCORE_FEATS
    N_tr  = train_df[num_cols].values;   N_te  = test_df[num_cols].values
    NS_tr = train_df[SCORE_FEATS].values; NS_te = test_df[SCORE_FEATS].values

    # TF-IDF + SVD  (fit solo en train)
    tfidf = TfidfVectorizer(ngram_range=(1, 3), min_df=3, max_df=1.0, sublinear_tf=False)
    T_tr  = tfidf.fit_transform(txt_tr)
    T_te  = tfidf.transform(txt_te)
    svd   = TruncatedSVD(n_components=30, random_state=42)
    S_tr  = svd.fit_transform(T_tr)
    S_te  = svd.transform(T_te)

    SC_tr  = np.hstack([S_tr, N_tr]);   SC_te  = np.hstack([S_te, N_te])
    SCS_tr = np.hstack([S_tr, NS_tr]);  SCS_te = np.hstack([S_te, NS_te])

    # ── CV paralelo (hilos, seguro con numpy) ────────────────────────────────
    # Cuando los experimentos ya corren en paralelo usamos pocos hilos aquí
    # para no saturar los núcleos; ajusta n_jobs_cv según tu hardware.
    n_jobs_cv = 2 if N_JOBS_EXPERIMENTS != 1 else 5
    best_msl  = best_min_samples_leaf(SC_tr, y_tr, n_splits=5, n_jobs_cv=n_jobs_cv)

    RF_P = dict(min_samples_leaf=best_msl)

    # ── Construir configs ────────────────────────────────────────────────────
    # Nota: se instancian aquí para que cada proceso tenga su propio objeto
    configs = [
        ("RF_best",       make_rf(**RF_P),      SC_tr,  SC_te),
        ("RF_txt",        make_rf(**RF_P),       S_tr,   S_te),
        ("RF_txt_scores", make_rf(**RF_P),       SCS_tr, SCS_te),
        ("GBR_txt",       make_gbr(),            S_tr,   S_te),
        ("GBR_txt_score", make_gbr(),            SCS_tr, SCS_te),
        ("GBR_all",       make_gbr(),            SC_tr,  SC_te),
        ("Ridge_txt",     make_ridge(),          S_tr,   S_te),
        ("Ridge_all",     make_ridge(),          SC_tr,  SC_te),
        ("SVR_txt",       make_svr_pipeline(),   S_tr,   S_te),
        ("SVR_all",       make_svr_pipeline(),   SC_tr,  SC_te),
    ]

    diff_te  = test_df[NOTA_DIFFICULTY_COL].values if APPLY_NOTA else None
    nota_gt  = compute_nota(y_te, diff_te) if APPLY_NOTA else y_te

    rows = []
    for mname, m, Xtr, Xte in configs:
        p = m.fit(Xtr, y_tr).predict(Xte)
        if GPU_AVAILABLE:
            p = np.asarray(p)           # cuML → numpy
        rho, _      = spearmanr(y_te, p)
        nota_pred   = compute_nota(p, diff_te) if APPLY_NOTA else p
        rho_nota, _ = spearmanr(nota_gt, nota_pred) if APPLY_NOTA else (rho, None)

        rows.append({
            "group_size":    group_size,
            "combination":   pair_label,
            "model":         mname,
            "spearman":      rho,
            "spearman_nota": rho_nota,
        })

    best_row = max(rows, key=lambda x: x["spearman_nota"])
    print(
        f"[{exp_idx+1:>3}/{total}] {pair_label[:55]:55s} "
        f"| mejor={best_row['model']:20s} ρ_nota={best_row['spearman_nota']:.4f} "
        f"t={time.time()-t0:.0f}s",
        flush=True,
    )
    return rows

# ── Ejecución paralela de todos los experimentos ─────────────────────────────
print(f"\n[RUN] Lanzando {total_experiments} experimentos "
      f"(n_jobs={N_JOBS_EXPERIMENTS}, GPU={'sí' if GPU_AVAILABLE else 'no'})...\n",
      flush=True)

# prefer="processes" aísla memoria entre workers (necesario con sklearn).
# Si tienes GPU y usas cuML, cambia a prefer="threads" para compartir contexto CUDA.
prefer = "threads" if GPU_AVAILABLE else "processes"

all_results_nested = Parallel(n_jobs=N_JOBS_EXPERIMENTS, prefer=prefer)(
    delayed(run_experiment)(vlm_subset, i, total_experiments)
    for i, vlm_subset in enumerate(experiments)
)

# Aplanar lista de listas
all_results = [row for rows in all_results_nested for row in rows]

# ── Resumen y guardado ────────────────────────────────────────────────────────
df_results = pd.DataFrame(all_results)
df_results.to_csv(OUTPUT_CSV, index=False, sep=";")
print(f"\nResultados guardados en: {OUTPUT_CSV}")

best_per_combo = (
    df_results
    .loc[df_results.groupby("combination")["spearman_nota"].idxmax()]
    .sort_values(["group_size", "spearman_nota"], ascending=[True, False])
    .reset_index(drop=True)
)

print("\n" + "="*80)
print(f"  {'Sz':>2}  {'Combinación':45s}  {'Modelo':20s}  Spearman_nota  Spearman")
print("-"*80)
for _, row in best_per_combo.iterrows():
    print(f"  {int(row['group_size']):>2}  {row['combination']:45s}  "
          f"{row['model']:20s}  {row['spearman_nota']:.4f}         {row['spearman']:.4f}")
print("="*80)

best_global = best_per_combo.loc[best_per_combo["spearman_nota"].idxmax()]
print(f"\n🏆 Mejor combinación global:")
print(f"   {best_global['combination']}  →  {best_global['model']}  "
      f"Spearman_nota={best_global['spearman_nota']:.4f}  Spearman={best_global['spearman']:.4f}")

print("\n📊 Mejor por tamaño de grupo:")
for size, grp in best_per_combo.groupby("group_size"):
    row = grp.loc[grp["spearman_nota"].idxmax()]
    print(f"   size={size}: {row['combination']}  →  {row['model']}  "
          f"Spearman_nota={row['spearman_nota']:.4f}  Spearman={row['spearman']:.4f}")

print(f"\nTiempo total: {time.time()-t0:.1f}s")