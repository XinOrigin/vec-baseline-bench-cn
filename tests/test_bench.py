"""Unit and end-to-end tests for vec-baseline-bench-cn."""
from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pytest

from vec_bench.baselines import (
    BASELINE_REGISTRY,
    CopyLastBaseline,
    LatentAffineTransportBaseline,
    PseudobulkShiftBaseline,
    SpatialGrowthTransportBaseline,
)
from vec_bench.benchmark import run_benchmark, split_disjoint_cells
from vec_bench.cli import main as cli_main
from vec_bench.report_html import render_html_report

VECKIT_DIR = Path(__file__).resolve().parents[2] / "veckit"


def _make_synthetic_pair(seed: int = 0, n_cells: int = 120, n_genes: int = 80) -> tuple[ad.AnnData, ad.AnnData]:
    rng = np.random.default_rng(seed)
    base_rates = rng.uniform(0.3, 2.5, size=(1, n_genes))
    x_ref = np.log1p(rng.poisson(base_rates * 4.0, size=(n_cells, n_genes)).astype(np.float32))
    # True developmental transition: 15 upregulated genes, 15 downregulated genes, plus 3D anisotropic growth
    shift = np.zeros(n_genes, dtype=np.float32)
    shift[:15] = 1.4
    shift[15:30] = -0.8
    x_tgt = np.maximum(
        np.log1p(rng.poisson(base_rates * 4.0, size=(n_cells, n_genes)).astype(np.float32)) + shift,
        0.0,
    ).astype(np.float32)

    coords_ref = rng.normal(loc=[0.0, 0.0, 0.0], scale=[10.0, 12.0, 8.0], size=(n_cells, 3)).astype(np.float32)
    coords_tgt = rng.normal(loc=[15.0, -5.0, 4.0], scale=[18.0, 14.0, 11.0], size=(n_cells, 3)).astype(np.float32)

    genes = [f"Gene_{i}" for i in range(n_genes)]
    cts = [f"ct_{i % 3}" for i in range(n_cells)]

    ref = ad.AnnData(X=x_ref)
    ref.var_names = genes
    ref.obs["celltype"] = cts
    ref.obsm["spatial_3D"] = coords_ref

    tgt = ad.AnnData(X=x_tgt)
    tgt.var_names = genes
    tgt.obs["celltype"] = cts
    tgt.obsm["spatial_3D"] = coords_tgt
    return ref, tgt


def test_disjoint_split_has_zero_overlap():
    ref, _ = _make_synthetic_pair(seed=1, n_cells=100, n_genes=20)
    ref.obs_names = [f"cell_{i}" for i in range(100)]
    tr, ev = split_disjoint_cells(ref, train_frac=0.6, seed=42)
    assert tr.n_obs == 60
    assert ev.n_obs == 40
    assert set(tr.obs_names).isdisjoint(set(ev.obs_names))


def test_all_baselines_preserve_validity_and_improve_synthetic_shift():
    ref, tgt = _make_synthetic_pair(seed=2, n_cells=120, n_genes=60)
    ref_tr, ref_ev = split_disjoint_cells(ref, train_frac=0.5, seed=0)
    tgt_tr, tgt_ev = split_disjoint_cells(tgt, train_frac=0.5, seed=1)

    floor_err = float(np.mean((ref_ev.X.mean(axis=0) - tgt_ev.X.mean(axis=0)) ** 2))
    floor_spatial_err = float(
        np.linalg.norm(ref_ev.obsm["spatial_3D"].mean(axis=0) - tgt_ev.obsm["spatial_3D"].mean(axis=0))
    )

    for name, cls in BASELINE_REGISTRY.items():
        model = cls(seed=0).fit(ref_tr, tgt_tr)
        pred = model.predict(ref_ev)
        assert pred.shape == ref_ev.shape
        assert list(pred.var_names) == list(ref_ev.var_names)
        assert np.isfinite(pred.X).all()
        assert float(pred.X.min()) >= 0.0
        # Non-collapsed single-cell heterogeneity
        assert float(pred.X.std(axis=0).mean()) > 0.1
        assert "spatial_3D" in pred.obsm
        assert pred.obsm["spatial_3D"].shape == (ref_ev.n_obs, 3)

        if name != "copy_last":
            err = float(np.mean((pred.X.mean(axis=0) - tgt_ev.X.mean(axis=0)) ** 2))
            assert err < floor_err * 0.25, f"{name} failed to reduce pseudobulk MSE"

        if name == "spatial_growth_ot":
            sp_err = float(
                np.linalg.norm(pred.obsm["spatial_3D"].mean(axis=0) - tgt_ev.obsm["spatial_3D"].mean(axis=0))
            )
            assert sp_err < floor_spatial_err * 0.4


@pytest.mark.skipif(not (VECKIT_DIR / "data" / "sample_wt.h5ad").exists(), reason="local veckit samples absent")
def test_end_to_end_benchmark_and_offline_html_on_real_veckit_t3(tmp_path: Path):
    data = VECKIT_DIR / "data"
    out_dir = tmp_path / "bench_t3"
    rc = cli_main([
        "run",
        "--task", "T3",
        "--ref", str(data / "sample_wt.h5ad"),
        "--target", str(data / "sample_mab21l2_ko.h5ad"),
        "--baselines", "copy_last,mean_shift,latent_ot,spatial_growth_ot",
        "--seed", "0",
        "--veckit-dir", str(VECKIT_DIR),
        "--out-dir", str(out_dir),
    ])
    assert rc == 0
    assert (out_dir / "report.md").exists()
    assert (out_dir / "report.json").exists()
    html_text = (out_dir / "report.html").read_text(encoding="utf-8")
    assert "<svg" in html_text
    assert "XY Projection" in html_text
    # Zero external CDN dependencies
    assert "https://cdn" not in html_text and "unpkg.com" not in html_text

    payload = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    by_name = {r["baseline"]: r for r in payload["results"]}
    assert by_name["copy_last"]["headline_skill_100"] == pytest.approx(50.0, abs=0.05)
    # Joint 3D spatial + expression OT beats both the 50.0 floor and expression-only latent_ot on T3
    assert by_name["spatial_growth_ot"]["headline_skill_100"] > 50.0
    assert by_name["spatial_growth_ot"]["headline_skill_100"] > by_name["latent_ot"]["headline_skill_100"]
    # Expression baselines still dramatically improve DE recovery and energy distance over copy_last
    assert by_name["mean_shift"]["raw_metrics"]["de_score"] > by_name["copy_last"]["raw_metrics"]["de_score"]
    assert by_name["mean_shift"]["raw_metrics"]["energy_distance"] < by_name["copy_last"]["raw_metrics"]["energy_distance"]


@pytest.mark.skipif(not (VECKIT_DIR / "data" / "sample_8.5.h5ad").exists(), reason="local veckit samples absent")
def test_end_to_end_t1_and_t2_beat_floor_without_leakage():
    data = VECKIT_DIR / "data"
    rep_t1 = run_benchmark(
        "T1",
        data / "sample_8.5.h5ad",
        data / "sample_9.5.h5ad",
        baseline_names=["copy_last", "mean_shift", "latent_ot"],
        seed=0,
        veckit_dir=VECKIT_DIR,
    )
    t1_by = {r["baseline"]: r for r in rep_t1["results"]}
    assert t1_by["copy_last"]["headline_skill_100"] == pytest.approx(50.0, abs=0.05)
    assert t1_by["mean_shift"]["headline_skill_100"] > 60.0
    assert t1_by["latent_ot"]["headline_skill_100"] > 60.0

    rep_t2 = run_benchmark(
        "T2",
        data / "sample_heart_9.25.h5ad",
        data / "sample_heart_9.5.h5ad",
        baseline_names=["copy_last", "mean_shift", "latent_ot", "spatial_growth_ot"],
        seed=0,
        auto_log1p=True,
        veckit_dir=VECKIT_DIR,
    )
    t2_by = {r["baseline"]: r for r in rep_t2["results"]}
    assert t2_by["copy_last"]["headline_skill_100"] == pytest.approx(50.0, abs=0.05)
    assert t2_by["latent_ot"]["headline_skill_100"] > 60.0
    assert t2_by["spatial_growth_ot"]["headline_skill_100"] > 50.0


@pytest.mark.skipif(not (VECKIT_DIR / "data" / "sample_8.5.h5ad").exists(), reason="local veckit samples absent")
def test_cli_predict_generates_valid_h5ad(tmp_path: Path):
    data = VECKIT_DIR / "data"
    out_h5ad = tmp_path / "pred_t1.h5ad"
    rc = cli_main([
        "predict",
        "--baseline", "mean_shift",
        "--train-ref", str(data / "sample_8.5.h5ad"),
        "--train-target", str(data / "sample_9.5.h5ad"),
        "--eval-ref", str(data / "sample_8.5.h5ad"),
        "--out", str(out_h5ad),
    ])
    assert rc == 0
    pred = ad.read_h5ad(out_h5ad)
    assert pred.shape == (150, 32285)
    assert float(pred.X.min()) >= 0.0
    assert np.isfinite(pred.X).all()


def test_run_benchmark_rejects_half_explicit_train_pair(tmp_path: Path):
    with pytest.raises(ValueError, match="passed together"):
        run_benchmark(
            "T2",
            tmp_path / "r.h5ad",
            tmp_path / "t.h5ad",
            train_ref_path=tmp_path / "x.h5ad",
        )
    with pytest.raises(ValueError, match="passed together"):
        run_benchmark(
            "T2",
            tmp_path / "r.h5ad",
            tmp_path / "t.h5ad",
            train_target_path=tmp_path / "y.h5ad",
        )


def test_rescale_uses_anchor_status_and_baseline_rejects_gene_mismatch():
    from vec_bench.baselines import make_baseline
    from vec_bench.benchmark import rescale_to_official_skill

    res = rescale_to_official_skill(
        "T1",
        {"de_score": 0.5, "mmd_u": 0.3},
        {"de_score": 0.2, "mmd_u": 0.6},
        {"de_score": 0.8, "mmd_u": 0.1},
    )
    assert res["per_metric"]["de_score"]["anchor_status"] == "ok"
    assert res["per_metric"]["mmd_u"]["anchor_status"] == "ok"

    a1 = ad.AnnData(X=np.ones((6, 3), dtype=np.float32))
    a1.var_names = ["G0", "G1", "G2"]
    a2 = ad.AnnData(X=np.ones((6, 4), dtype=np.float32))
    a2.var_names = ["G0", "G1", "G2", "G3"]
    base = make_baseline("mean_shift")
    with pytest.raises(ValueError, match="var_names"):
        base.fit(a1, a2)
