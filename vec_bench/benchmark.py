"""Leakage-free local benchmark runner integrating official aristoteleo/veckit scoring."""
from __future__ import annotations

import argparse
import math
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np

from .baselines import make_baseline, normalize_log1p_if_needed, to_dense_float32

METRIC_SPECS: dict[str, tuple[str, bool]] = {
    "de_score": ("higher", False),
    "de_direction": ("higher", False),
    "severity_slope": ("higher", False),
    "energy_distance": ("lower", False),
    "mmd_u": ("lower", False),
    "variogram": ("lower", False),
    "d2_shape": ("lower", False),
    "sliced_wasserstein": ("lower", False),
    "occupancy_dice": ("higher", False),
    "scale_log_ratio": ("lower", True),
    "count_log_ratio": ("lower", True),
    "neighborhood_mmd": ("lower", False),
}

TASK_GROUPS: dict[str, list[tuple[str, float, list[str]]]] = {
    "T1": [
        ("de_gene_recovery", 0.25, ["de_score"]),
        ("change_direction", 0.25, ["de_direction"]),
        ("cell_state_distribution", 0.30, ["mmd_u", "energy_distance"]),
        ("gene_gene_covariation", 0.20, ["variogram"]),
    ],
    "T2": [
        ("expression_de", 0.25, ["de_score", "de_direction"]),
        ("expression_distribution", 0.25, ["mmd_u", "energy_distance", "variogram"]),
        ("spatial_shape", 0.25, ["d2_shape", "sliced_wasserstein", "occupancy_dice"]),
        ("growth_and_local", 0.25, ["scale_log_ratio", "count_log_ratio", "neighborhood_mmd"]),
    ],
    "T3": [
        ("perturbation_response", 0.40, ["de_score", "de_direction", "severity_slope"]),
        ("mutant_distribution", 0.30, ["mmd_u", "variogram"]),
        ("spatial_shape", 0.30, ["d2_shape", "sliced_wasserstein", "occupancy_dice", "scale_log_ratio", "count_log_ratio", "neighborhood_mmd"]),
    ],
}


def load_veckit_scorer(veckit_dir: Path | None = None):
    if veckit_dir is not None:
        p = str(veckit_dir.resolve())
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        import score_h5ad  # type: ignore
        return score_h5ad
    except ImportError as exc:
        raise RuntimeError(
            "aristoteleo/veckit is required. Run `pip install git+https://github.com/aristoteleo/vec_baselines.git` "
            "or pass `--veckit-dir /path/to/veckit`."
        ) from exc


def score_with_veckit(
    score_mod: Any,
    task: str,
    pred_path: Path,
    target_path: Path,
    ref_path: Path,
    *,
    seed: int = 0,
    setting: str = "heart",
) -> dict[str, Any]:
    ns = argparse.Namespace(
        task=task,
        input=pred_path,
        target=target_path,
        reference=ref_path if task in {"T1", "T2"} else None,
        wt=ref_path if task == "T3" else None,
        seed=seed,
        allow_reorder=False,
        setting=setting,
    )
    if task == "T1":
        _, scores = score_mod.score_t1(ns)
    elif task == "T2":
        _, scores = score_mod.score_t2(ns)
    else:
        _, scores = score_mod.score_t3(ns)
    return scores


def rescale_to_official_skill(
    task: str,
    pred_metrics: dict[str, Any],
    floor_metrics: dict[str, Any],
    ceiling_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Official floor=50 / ceiling=100 rescaling with inverted-anchor guard for small subsamples."""
    per_metric: dict[str, dict[str, Any]] = {}
    inverted: list[str] = []
    for name, (direction, take_abs) in METRIC_SPECS.items():
        if name not in pred_metrics and name not in floor_metrics:
            continue
        v, vf, vc = pred_metrics.get(name), floor_metrics.get(name), ceiling_metrics.get(name)
        if v is None or vf is None or vc is None or not all(math.isfinite(x) for x in (v, vf, vc)):
            per_metric[name] = {"raw": v, "floor": vf, "ceiling": vc, "anchor_status": "missing", "skill_100": 0.0}
            continue
        m = abs(float(v)) if take_abs else float(v)
        mf = abs(float(vf)) if take_abs else float(vf)
        mc = abs(float(vc)) if take_abs else float(vc)

        is_inv = (direction == "lower" and mc > mf) or (direction == "higher" and mc < mf)
        if is_inv:
            inverted.append(name)
            per_metric[name] = {"raw": v, "floor": vf, "ceiling": vc, "anchor_status": "inverted", "skill_100": None}
            continue

        d_floor = abs(mc - mf)
        if d_floor <= 1e-12:
            better_or_eq = (m <= mc) if direction == "lower" else (m >= mc)
            sk = 1.0 if better_or_eq else 0.0
            status = "degenerate"
        else:
            d_m = (m - mc) if direction == "lower" else (mc - m)
            sk = 1.0 if d_m <= 0.0 else min(d_floor / (d_floor + d_m), 1.0)
            status = "ok"
        per_metric[name] = {
            "raw": v,
            "floor": vf,
            "ceiling": vc,
            "anchor_status": status,
            "skill_100": round(sk * 100.0, 2),
        }

    group_scores: dict[str, float | None] = {}
    headline = 0.0
    total_w = 0.0
    for gname, w, members in TASK_GROUPS[task]:
        vals = [per_metric[m]["skill_100"] for m in members if m in per_metric and per_metric[m]["skill_100"] is not None]
        if not vals:
            if any(m in per_metric for m in members):
                group_scores[gname] = None
            continue
        gs = sum(vals) / len(vals)
        group_scores[gname] = round(gs, 2)
        headline += w * gs
        total_w += w

    return {
        "headline_skill_100": round(headline / total_w, 2) if total_w > 0 else 0.0,
        "group_scores_100": group_scores,
        "per_metric": per_metric,
        "inverted_anchors": inverted,
    }


def split_disjoint_cells(adata: ad.AnnData, *, train_frac: float = 0.5, seed: int = 0) -> tuple[ad.AnnData, ad.AnnData]:
    """Deterministically split cells into two strictly disjoint train/eval AnnData subsets."""
    if not (0.1 <= train_frac <= 0.9):
        raise ValueError(f"train_frac must be in [0.1, 0.9], got {train_frac}")
    n = adata.n_obs
    if n < 8:
        raise ValueError(f"Need at least 8 cells for a disjoint train/eval split, got {n}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    cut = max(4, min(n - 4, int(round(n * train_frac))))
    return adata[perm[:cut]].copy(), adata[perm[cut:]].copy()


def _normalize_anndata_copy(adata: ad.AnnData, auto_log1p: bool) -> ad.AnnData:
    out = adata.copy()
    out.X = normalize_log1p_if_needed(to_dense_float32(out), auto_log1p=auto_log1p)
    return out


def run_benchmark(
    task: str,
    ref_path: Path,
    target_path: Path,
    *,
    train_ref_path: Path | None = None,
    train_target_path: Path | None = None,
    ceiling_target_path: Path | None = None,
    baseline_names: list[str] | None = None,
    seed: int = 0,
    train_frac: float = 0.5,
    auto_log1p: bool = False,
    setting: str = "heart",
    veckit_dir: Path | None = None,
) -> dict[str, Any]:
    """Run a leakage-free multi-baseline benchmark and return structured metrics + visualization data."""
    task = task.upper()
    if task not in TASK_GROUPS:
        raise ValueError(f"Unsupported task {task!r}")
    if (train_ref_path is None) != (train_target_path is None):
        raise ValueError("train_ref_path and train_target_path must be passed together")

    score_mod = load_veckit_scorer(veckit_dir)
    if baseline_names is None:
        baseline_names = ["copy_last", "mean_shift", "latent_ot"] + (
            ["spatial_growth_ot"] if task in {"T2", "T3"} else []
        )
    if "copy_last" not in baseline_names:
        baseline_names = ["copy_last", *baseline_names]

    eval_ref_raw = ad.read_h5ad(ref_path)
    eval_tgt_raw = ad.read_h5ad(target_path)

    if train_ref_path is not None and train_target_path is not None:
        split_mode = "explicit_train_eval"
        ref_train = _normalize_anndata_copy(ad.read_h5ad(train_ref_path), auto_log1p)
        tgt_train = _normalize_anndata_copy(ad.read_h5ad(train_target_path), auto_log1p)
        ref_eval = _normalize_anndata_copy(eval_ref_raw, auto_log1p)
        tgt_eval = _normalize_anndata_copy(eval_tgt_raw, auto_log1p)
    else:
        split_mode = f"disjoint_cell_split_{int(round(train_frac * 100))}_{int(round((1 - train_frac) * 100))}"
        ref_train, ref_eval = split_disjoint_cells(_normalize_anndata_copy(eval_ref_raw, auto_log1p), train_frac=train_frac, seed=seed)
        tgt_train, tgt_eval = split_disjoint_cells(_normalize_anndata_copy(eval_tgt_raw, auto_log1p), train_frac=train_frac, seed=seed + 1)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        eval_ref_file = tmp / "eval_ref.h5ad"
        eval_tgt_file = tmp / "eval_tgt.h5ad"
        ref_eval.write_h5ad(eval_ref_file)
        tgt_eval.write_h5ad(eval_tgt_file)

        # Compute attainable ceiling (explicit biological replicate or disjoint split-half of eval target)
        if ceiling_target_path is not None:
            ceil_mode = "replicate"
            ceil_ad = _normalize_anndata_copy(ad.read_h5ad(ceiling_target_path), auto_log1p)
            ceil_file = tmp / "ceil_rep.h5ad"
            ceil_ad.write_h5ad(ceil_file)
            ceiling_metrics = score_with_veckit(
                score_mod, task, ceil_file, eval_tgt_file, eval_ref_file, seed=seed, setting=setting
            )
        else:
            ceil_mode = "split_half"
            h_a, h_b = split_disjoint_cells(tgt_eval, train_frac=0.5, seed=seed + 7)
            ha_file, hb_file = tmp / "half_a.h5ad", tmp / "half_b.h5ad"
            h_a.write_h5ad(ha_file)
            h_b.write_h5ad(hb_file)
            ceiling_metrics = score_with_veckit(
                score_mod, task, ha_file, hb_file, eval_ref_file, seed=seed, setting=setting
            )

        raw_by_name: dict[str, dict[str, Any]] = {}
        elapsed_by_name: dict[str, float] = {}
        desc_by_name: dict[str, str] = {}
        viz_clouds: dict[str, list[list[float]]] = {}

        if "spatial_3D" in ref_eval.obsm and "spatial_3D" in tgt_eval.obsm:
            viz_clouds["reference"] = np.asarray(ref_eval.obsm["spatial_3D"], dtype=float).round(3).tolist()
            viz_clouds["target"] = np.asarray(tgt_eval.obsm["spatial_3D"], dtype=float).round(3).tolist()

        for bname in baseline_names:
            model = make_baseline(bname, seed=seed, auto_log1p=False)  # inputs already normalized if auto_log1p
            t0 = time.perf_counter()
            model.fit(ref_train, tgt_train)
            pred_ad = model.predict(ref_eval)
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)

            pred_file = tmp / f"pred_{bname}.h5ad"
            pred_ad.write_h5ad(pred_file)
            raw_metrics = score_with_veckit(
                score_mod, task, pred_file, eval_tgt_file, eval_ref_file, seed=seed, setting=setting
            )
            raw_by_name[bname] = raw_metrics
            elapsed_by_name[bname] = elapsed_ms
            desc_by_name[bname] = model.description
            if "spatial_3D" in pred_ad.obsm:
                viz_clouds[bname] = np.asarray(pred_ad.obsm["spatial_3D"], dtype=float).round(3).tolist()

    floor_metrics = raw_by_name["copy_last"]
    results: list[dict[str, Any]] = []
    for bname in baseline_names:
        skill = rescale_to_official_skill(task, raw_by_name[bname], floor_metrics, ceiling_metrics)
        results.append({
            "baseline": bname,
            "description": desc_by_name[bname],
            "elapsed_ms": elapsed_by_name[bname],
            "headline_skill_100": skill["headline_skill_100"],
            "delta_vs_floor": round(skill["headline_skill_100"] - 50.0, 2),
            "group_scores_100": skill["group_scores_100"],
            "per_metric": skill["per_metric"],
            "inverted_anchors": skill["inverted_anchors"],
            "raw_metrics": raw_by_name[bname],
        })

    results.sort(key=lambda r: r["headline_skill_100"], reverse=True)
    return {
        "task": task,
        "split_mode": split_mode,
        "ceiling_mode": ceil_mode,
        "auto_log1p": auto_log1p,
        "shapes": {
            "ref_train": list(ref_train.shape),
            "tgt_train": list(tgt_train.shape),
            "ref_eval": list(ref_eval.shape),
            "tgt_eval": list(tgt_eval.shape),
        },
        "results": results,
        "spatial_clouds": viz_clouds,
    }


def render_markdown_table(report: dict[str, Any]) -> str:
    lines = [
        f"## Virtual Embryo Challenge — {report['task']} Local Baseline Benchmark",
        "",
        f"- **Split mode:** `{report['split_mode']}` (zero train/eval cell overlap)",
        f"- **Ceiling mode:** `{report['ceiling_mode']}` | **auto_log1p:** `{report['auto_log1p']}`",
        f"- **Shapes:** train ref `{report['shapes']['ref_train']}` / target `{report['shapes']['tgt_train']}`, "
        f"eval ref `{report['shapes']['ref_eval']}` / target `{report['shapes']['tgt_eval']}`",
        "",
        "| Rank | Baseline | Headline Skill (0–100) | vs Floor (50) | DE Score | Energy Dist | MMD | Time (ms) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(report["results"], 1):
        rm = row["raw_metrics"]
        fmt = lambda k: f"{rm[k]:.4f}" if isinstance(rm.get(k), (int, float)) else "—"
        delta = f"{row['delta_vs_floor']:+.2f}"
        lines.append(
            f"| {rank} | `{row['baseline']}` | **{row['headline_skill_100']:.2f}** | {delta} | "
            f"{fmt('de_score')} | {fmt('energy_distance')} | {fmt('mmd_u')} | {row['elapsed_ms']:.1f} |"
        )
    inv = report["results"][0].get("inverted_anchors") if report["results"] else []
    if inv:
        lines.extend([
            "",
            f"> **Note:** Split-half ceiling anchor was inverted on small subsample for `{', '.join(inv)}` "
            "(excluded from headline skill; evaluate on full embryos or pass `--ceiling-target`).",
        ])
    return "\n".join(lines) + "\n"
