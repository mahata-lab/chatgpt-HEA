"""Score the fresh rerun without unstable or artificially favorable metrics.

Fixes relative to the uploaded script:
- deterministic majority vote; missing/UNK runs reduce agreement,
- per-tier R2 is NA when the reference has no variance,
- macro-F1 is averaged over classes actually present in the tier's reference set,
- pooled R2 uses one GLOBAL reference standardization per descriptor,
- reproducibility uses run-to-run SD of dH (kJ/mol), NOT CV/|mean|, which explodes near zero.
"""
from collections import Counter
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    r2_score, mean_squared_error, accuracy_score,
    precision_recall_fscore_support,
)

import hea_descriptors as hd
from config import DATA_DIR, TIERS, PHASE_CLASSES, N_RUNS

TIER_ORDER = list(TIERS.keys())
DESCS = ["dS", "dH", "VEC", "delta"]


def majority_phase(series):
    vals = [str(x) for x in series if str(x) in PHASE_CLASSES]
    if not vals:
        return "UNK", 0.0
    counts = Counter(vals)
    max_n = max(counts.values())
    winners = sorted([k for k, v in counts.items() if v == max_n])
    # A tie is genuine ambiguity, not an arbitrary set-order choice.
    phase = winners[0] if len(winners) == 1 else "UNK"
    agreement = (max_n / N_RUNS) * 100.0
    return phase, agreement


def metric_row(tier, key, d):
    y = d.reference.to_numpy(float)
    p = d.predicted.to_numpy(float)
    rmse = float(np.sqrt(mean_squared_error(y, p)))
    mae = float(np.mean(np.abs(y - p)))
    nz = np.abs(y) > 1e-9
    mape = float(np.mean(np.abs((y[nz] - p[nz]) / y[nz])) * 100.0) if nz.any() else np.nan
    # R2 is undefined when y_true is constant. Do not let sklearn force it to 0/1.
    r2 = float(r2_score(y, p)) if np.var(y) > 1e-12 else np.nan
    return {"tier": tier, "descriptor": key, "n": len(d),
            "RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}


def main():
    runs_path = DATA_DIR / "runs.csv"
    if not runs_path.exists():
        raise SystemExit(f"Missing {runs_path}. Run query_chatgpt.py first.")
    runs = pd.read_csv(runs_path)

    # Strict integrity checks: exactly five unique runs per system.
    problems = []
    for tier, systems in TIERS.items():
        for system in systems:
            g = runs[(runs.tier == tier) & (runs.system == system)]
            if len(g) != N_RUNS or g.run.nunique() != N_RUNS:
                problems.append(f"{tier}/{system}: rows={len(g)}, unique runs={g.run.nunique()}")
    if problems:
        raise SystemExit("Run table is incomplete/duplicated:\n" + "\n".join(problems))

    ref = pd.read_csv("reference_phases.csv")
    required = {"system", "reference_phase"}
    if not required.issubset(ref.columns):
        raise SystemExit("reference_phases.csv must contain system,reference_phase")
    ref_ph = ref.set_index("system")["reference_phase"].to_dict()
    missing_ref = [s for systems in TIERS.values() for s in systems if s not in ref_ph]
    if missing_ref:
        raise SystemExit(f"Missing phase references for: {missing_ref}")
    print("WARNING: phase labels are only as defensible as reference_phases.csv. Verify each label against cited literature/CALPHAD before publication.")

    agg_rows, repro_rows = [], []
    for (tier, system), g in runs.groupby(["tier", "system"], sort=False):
        gt = hd.all_descriptors(system)
        rec = {"tier": tier, "system": system}
        for k in DESCS:
            vals = pd.to_numeric(g[k], errors="coerce").dropna().to_numpy(float)
            rec[f"pred_{k}"] = float(np.mean(vals)) if len(vals) else np.nan
            rec[f"sd_{k}"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
            rec[f"ref_{k}"] = gt[k]
            rec[f"n_valid_{k}"] = int(len(vals))
        phase, agree = majority_phase(g.phase)
        rec["pred_phase"] = phase
        rec["ref_phase"] = str(ref_ph.get(system, "UNK")).upper().strip()
        rec["phase_agreement_pct"] = agree
        rec["n_valid_phase"] = int(sum(str(x) in PHASE_CLASSES for x in g.phase))
        agg_rows.append(rec)
        repro_rows.append({
            "tier": tier, "system": system,
            "SD_dH_kJmol": rec["sd_dH"],
            "SD_delta_pctpoint": rec["sd_delta"],
            "categorical_agreement_pct": agree,
            "valid_phase_runs_pct": 100.0 * rec["n_valid_phase"] / N_RUNS,
        })

    agg = pd.DataFrame(agg_rows)
    agg.to_csv(DATA_DIR / "aggregated_by_system.csv", index=False)

    long_rows = []
    for _, r in agg.iterrows():
        for k in DESCS:
            long_rows.append({"system": r.system, "tier": r.tier, "descriptor": k,
                              "reference": r[f"ref_{k}"], "predicted": r[f"pred_{k}"]})
    long = pd.DataFrame(long_rows).dropna(subset=["predicted"])
    long.to_csv(DATA_DIR / "data_numeric_predictions.csv", index=False)

    rows = []
    for tier in TIER_ORDER:
        for k in DESCS:
            d = long[(long.tier == tier) & (long.descriptor == k)]
            if len(d) >= 2:
                rows.append(metric_row(tier, k, d))
    # Also report overall by descriptor.
    for k in DESCS:
        d = long[long.descriptor == k]
        rows.append(metric_row("Overall", k, d))
    pd.DataFrame(rows).to_csv(DATA_DIR / "metrics_numeric.csv", index=False)

    # Pooled standardized R2. Standardization is learned ONCE from all 24 reference
    # values for each descriptor, not separately within each tier.
    scale = {}
    for k in DESCS:
        dk = long[long.descriptor == k]
        mu = float(dk.reference.mean())
        sd = float(dk.reference.std(ddof=0))
        if sd <= 1e-12:
            raise SystemExit(f"Descriptor {k} has zero global reference variance")
        scale[k] = (mu, sd)
    pooled_rows = []
    for tier in TIER_ORDER + ["Overall"]:
        d = long if tier == "Overall" else long[long.tier == tier]
        zr, zp = [], []
        for _, r in d.iterrows():
            mu, sd = scale[r.descriptor]
            zr.append((r.reference - mu) / sd)
            zp.append((r.predicted - mu) / sd)
        pooled_rows.append({
            "tier": tier, "n_points": len(zr),
            "pooled_R2_global_standardization": float(r2_score(zr, zp)),
            "pooled_RMSE_z": float(np.sqrt(mean_squared_error(zr, zp))),
        })
    pd.DataFrame(pooled_rows).to_csv(DATA_DIR / "metrics_numeric_pooled.csv", index=False)

    # Classification. Accuracy counts UNK as wrong. Macro metrics average over
    # reference classes present in that tier, avoiding artificial zeros from absent classes.
    cls_rows = []
    cm = np.zeros((len(PHASE_CLASSES), len(PHASE_CLASSES) + 1), dtype=int)
    pred_cols = PHASE_CLASSES + ["UNK"]
    ridx = {c: i for i, c in enumerate(PHASE_CLASSES)}
    pidx = {c: i for i, c in enumerate(pred_cols)}
    valid_ref = agg[agg.ref_phase.isin(PHASE_CLASSES)].copy()
    for _, r in valid_ref.iterrows():
        cm[ridx[r.ref_phase], pidx.get(r.pred_phase, pidx["UNK"])] += 1

    for tier in TIER_ORDER + ["Overall"]:
        d = valid_ref if tier == "Overall" else valid_ref[valid_ref.tier == tier]
        if d.empty:
            continue
        y_true = d.ref_phase.astype(str).to_numpy()
        y_pred = d.pred_phase.astype(str).to_numpy()
        labels = [c for c in PHASE_CLASSES if c in set(y_true)]
        acc = float(accuracy_score(y_true, y_pred))
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        )
        cls_rows.append({
            "tier": tier, "n": len(d), "classes_in_reference": "|".join(labels),
            "accuracy": acc, "macro_precision": float(precision),
            "macro_recall": float(recall), "macro_F1": float(f1),
            "unknown_predictions": int(np.sum(y_pred == "UNK")),
        })
    pd.DataFrame(cls_rows).to_csv(DATA_DIR / "metrics_classification.csv", index=False)
    pd.DataFrame(cm, index=PHASE_CLASSES, columns=pred_cols).to_csv(DATA_DIR / "confusion_matrix.csv")

    repro = pd.DataFrame(repro_rows)
    repro.to_csv(DATA_DIR / "reproducibility_by_system.csv", index=False)
    tier_repro = (repro.groupby("tier", sort=False)
                  .agg(mean_SD_dH_kJmol=("SD_dH_kJmol", "mean"),
                       median_SD_dH_kJmol=("SD_dH_kJmol", "median"),
                       mean_SD_delta_pctpoint=("SD_delta_pctpoint", "mean"),
                       categorical_agreement_pct=("categorical_agreement_pct", "mean"),
                       valid_phase_runs_pct=("valid_phase_runs_pct", "mean"))
                  .reindex(TIER_ORDER).reset_index())
    tier_repro.to_csv(DATA_DIR / "metrics_reproducibility.csv", index=False)

    summary = {
        "warning": "Use these outputs as the source of truth. Do not tune prompts or delete unfavorable outcomes after viewing them.",
        "data_dir": str(DATA_DIR),
    }
    (DATA_DIR / "scoring_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nNumeric metrics:\n", pd.read_csv(DATA_DIR / "metrics_numeric.csv").to_string(index=False))
    print("\nPooled metrics:\n", pd.read_csv(DATA_DIR / "metrics_numeric_pooled.csv").to_string(index=False))
    print("\nClassification:\n", pd.read_csv(DATA_DIR / "metrics_classification.csv").to_string(index=False))
    print("\nReproducibility:\n", tier_repro.to_string(index=False))


if __name__ == "__main__":
    main()
