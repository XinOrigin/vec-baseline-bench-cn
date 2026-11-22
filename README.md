# vec-baseline-bench-cn — Virtual Embryo Challenge (NeurIPS 2026) 轻量 CPU 基线库 & 无泄漏离线评测仪

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **CPU-only Starter Baselines, Leakage-Free Local Benchmark Harness, and Zero-CDN Offline 3D Spatial Report Generator** for the [Virtual Embryo Challenge (NeurIPS 2026)](https://virtualembryo.ai/challenge).
>
> Companion to [`XinOrigin/vec-preflight-cn`](https://github.com/XinOrigin/vec-preflight-cn): while `vec-preflight-cn` validates `.h5ad` submissions before upload, **`vec-baseline-bench-cn`** helps teams without GPUs move cleanly beyond the trivial `copy_last` / `wt_identity` floor (`50.0`) with transparent, reproducible CPU baselines and side-by-side 3D spatial diagnostics.

📖 **完整中文基线原理与进阶实战指南**: [`docs/BASELINES_ZH.md`](docs/BASELINES_ZH.md)

---

## Included CPU-Only Baselines / 内置零 GPU 进阶基线

Every baseline implements a clean `.fit(ref_train, target_train)` → `.predict(ref_eval)` interface, runs in **milliseconds on CPU**, and preserves exact `var_names`, non-negative `float32` expression, single-cell heterogeneity, and `obsm["spatial_3D"]`:

| Baseline | Core Idea | Target Tasks | Why it beats naive baselines |
|---|---|---|---|
| `copy_last` | Official identity floor (`50.0`) | T1 / T2 / T3 | Required anchor for floor/ceiling rescaling |
| `mean_shift` | Variance-preserving L2-shrunk pseudobulk + celltype-stratified shift | T1 / T2 / T3 | Avoids the classic "predict the mean cell" variance collapse that destroys `mmd_u` & `variogram` |
| `latent_ot` | Closed-form Bures-Wasserstein (Gaussian OT) in truncated SVD space + orthogonal residual preservation | T1 / T2 / T3 | Aligns both 1st moment (DE recovery) and 2nd moment (gene-gene covariance & cell distribution) |
| `spatial_growth_ot` | Joint 3D Gaussian-OT morphogenesis map on `obsm['spatial_3D']` + `latent_ot` expression transport | **T2 / T3** | Explicitly transforms 3D organ geometry (`d2_shape`, `sliced_wasserstein`, `occupancy_dice`, `scale_log_ratio`) alongside expression |

---

## Verified Zero-Leakage Results on `aristoteleo/veckit/data/`

Measured with `vec-bench run` using strict **50/50 disjoint cell splits** (zero cell overlap between `fit()` and `predict()`/`score()`), scored via official [`aristoteleo/veckit`](https://github.com/aristoteleo/veckit) on an Apple M4 CPU:

### Task 1 (`sample_8.5.h5ad` → `sample_9.5.h5ad`, 32,285 genes)
| Rank | Baseline | Headline Skill (0–100) | vs Floor (50) | DE Score | Energy Dist | MMD | CPU Time |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `mean_shift` | **65.77** | +15.77 | 0.6667 | 0.4414 | 0.1316 | 57.2 ms |
| 2 | `latent_ot` | **64.10** | +14.10 | 0.6667 | 1.2343 | **0.1206** | 109.3 ms |
| 3 | `copy_last` | **50.00** | +0.00 | 0.0000 | 0.2865 | 0.1278 | 7.9 ms |

### Task 2 (`sample_heart_9.25.h5ad` → `sample_heart_9.5.h5ad`, 500 genes + 3D spatial, `--auto-log1p`)
| Rank | Baseline | Headline Skill (0–100) | vs Floor (50) | DE Score | Energy Dist | MMD | CPU Time |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `latent_ot` | **63.92** | +13.92 | 0.3810 | 1.3877 | **0.0940** | 3.8 ms |
| 2 | `mean_shift` | **63.65** | +13.65 | 0.3810 | **1.1006** | 0.1035 | 3.4 ms |
| 3 | `spatial_growth_ot` | **55.80** | +5.80 | 0.3810 | 1.3877 | 0.0940 | 5.5 ms |
| 4 | `copy_last` | **50.00** | +0.00 | 0.0000 | 2.1494 | 0.1155 | 1.0 ms |

### Task 3 (`sample_wt.h5ad` → `sample_mab21l2_ko.h5ad`, 500 genes + 3D KO spatial)
| Rank | Baseline | Headline Skill (0–100) | vs Floor (50) | DE Score | Energy Dist | MMD | CPU Time |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `spatial_growth_ot` | **56.97** | **+6.97** | 0.3636 | 0.6363 | **0.0643** | 4.0 ms |
| 2 | `copy_last` | **50.00** | +0.00 | 0.0000 | 1.1313 | 0.0728 | 0.9 ms |
| 3 | `mean_shift` | **48.16** | -1.84 | **0.4545** | **0.3007** | 0.0644 | 1.6 ms |
| 4 | `latent_ot` | **46.44** | -3.56 | 0.3636 | 0.6363 | 0.0643 | 4.3 ms |

> **Key Scientific Takeaway from T3:** Expression-only `latent_ot` and `mean_shift` improve `de_score` and `energy_distance` substantially, yet adding the 3D spatial Gaussian-OT growth map (`spatial_growth_ot`) lifts T3 headline skill by **+10.53 points** over expression-only `latent_ot` — demonstrating why T2/T3 models must jointly transform `obsm['spatial_3D']` rather than leaving reference geometry static.

---

## Quickstart / 快速上手

```bash
git clone https://github.com/XinOrigin/vec-baseline-bench-cn.git
cd vec-baseline-bench-cn
pip install -e .
# 国内镜像加速: pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -e .
pip install git+https://github.com/aristoteleo/vec_baselines.git  # or pass --veckit-dir /path/to/veckit
```

### 1. Run a leakage-free multi-baseline benchmark + offline 3D HTML report

```bash
# Single pair (automatic 50/50 disjoint train/eval split)
vec-bench run \
  --task T3 \
  --ref /path/to/sample_wt.h5ad \
  --target /path/to/sample_mab21l2_ko.h5ad \
  --out-dir ./report_t3

# Full training data: explicit separate training & validation embryos (recommended!)
vec-bench run \
  --task T2 \
  --train-ref embryo1_E9.25.h5ad \
  --train-target embryo1_E9.5.h5ad \
  --ref embryo2_E9.25.h5ad \
  --target embryo2_E9.5.h5ad \
  --ceiling-target embryo3_E9.5.h5ad \
  --out-dir ./report_t2
```

Open `./report_t3/report.html` directly in any browser — **zero external CDN dependencies**, includes inline SVG **XY / XZ / YZ 3D spatial projections** comparing Reference, Held-out Target, and Predicted morphology.

### 2. Generate a valid `.h5ad` prediction file directly

```bash
vec-bench predict \
  --baseline spatial_growth_ot \
  --train-ref train_wt.h5ad \
  --train-target train_ko.h5ad \
  --eval-ref test_wt.h5ad \
  --out pred_t3.h5ad
```

Then pre-check `pred_t3.h5ad` with [`vec-preflight-cn`](https://github.com/XinOrigin/vec-preflight-cn) before using a scarce P3 submission slot!

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest -v
```

---

## License

MIT — see [`LICENSE`](LICENSE). Independent community resource for the Virtual Embryo Challenge; not affiliated with Stanford Qiu Lab.
