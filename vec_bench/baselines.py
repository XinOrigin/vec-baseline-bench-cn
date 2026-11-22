"""Lightweight, CPU-only starter baselines for Virtual Embryo Challenge T1, T2, and T3.

Every baseline adheres strictly to a fit(ref_train, target_train) -> predict(ref_eval)
contract so local benchmarks remain leakage-free when run on disjoint cell splits or
separate training/validation embryos.

All predicted AnnData objects preserve:
  - exact gene panel and gene column order (`var_names`)
  - finite, non-negative float32 expression matrix (`X`)
  - non-collapsed single-cell population variance
  - `obsm["spatial_3D"]` (shape `[n_cells, 3]`, float32) whenever spatial coordinates exist
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import anndata as ad
import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD


def to_dense_float32(adata: ad.AnnData) -> np.ndarray:
    x = adata.X
    if sparse.issparse(x):
        return np.asarray(x.toarray(), dtype=np.float32)
    return np.asarray(x, dtype=np.float32).copy()


def looks_like_raw_counts(x: np.ndarray) -> bool:
    nz = x[x > 0]
    if nz.size == 0:
        return False
    int_frac = float(np.mean(np.isclose(nz, np.round(nz), atol=1e-4)))
    return float(np.max(x)) > 20.0 and int_frac > 0.90


def normalize_log1p_if_needed(
    x: np.ndarray,
    *,
    target_sum: float = 1e4,
    auto_log1p: bool = False,
) -> np.ndarray:
    """Optionally apply standard library-size + log1p normalization when raw counts are detected."""
    if not auto_log1p or not looks_like_raw_counts(x):
        return x
    row_sums = np.sum(x, axis=1, keepdims=True)
    safe_sums = np.where(row_sums > 0, row_sums, 1.0)
    scaled = x / safe_sums * target_sum
    return np.log1p(np.maximum(scaled, 0.0)).astype(np.float32)


def _align_expression(ref: ad.AnnData, target: ad.AnnData, auto_log1p: bool) -> tuple[np.ndarray, np.ndarray]:
    if list(ref.var_names) != list(target.var_names):
        raise ValueError(
            "Reference and target AnnData objects must share identical var_names in the same order."
        )
    x_ref = normalize_log1p_if_needed(to_dense_float32(ref), auto_log1p=auto_log1p)
    x_tgt = normalize_log1p_if_needed(to_dense_float32(target), auto_log1p=auto_log1p)
    return x_ref, x_tgt


def _build_output_adata(
    template: ad.AnnData,
    x_pred: np.ndarray,
    coords_pred: np.ndarray | None = None,
) -> ad.AnnData:
    x_clean = np.nan_to_num(np.asarray(x_pred, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    x_clean = np.maximum(x_clean, 0.0)
    out = ad.AnnData(X=x_clean, obs=template.obs.copy(), var=template.var.copy())
    if coords_pred is not None:
        out.obsm["spatial_3D"] = np.asarray(coords_pred, dtype=np.float32)
    elif "spatial_3D" in template.obsm:
        out.obsm["spatial_3D"] = np.asarray(template.obsm["spatial_3D"], dtype=np.float32).copy()
    return out


class BaseEmbryoBaseline(ABC):
    name: str = "base"
    description: str = ""

    def __init__(self, *, seed: int = 0, auto_log1p: bool = False) -> None:
        self.seed = seed
        self.auto_log1p = auto_log1p

    @abstractmethod
    def fit(self, ref_train: ad.AnnData, target_train: ad.AnnData) -> "BaseEmbryoBaseline":
        ...

    @abstractmethod
    def predict(self, ref_eval: ad.AnnData) -> ad.AnnData:
        ...


class CopyLastBaseline(BaseEmbryoBaseline):
    """Official floor anchor (`copy_last` for T1/T2, `wt_identity` for T3)."""

    name = "copy_last"
    description = "Identity floor baseline returning the reference stage (or matched WT) unchanged."

    def fit(self, ref_train: ad.AnnData, target_train: ad.AnnData) -> "CopyLastBaseline":
        return self

    def predict(self, ref_eval: ad.AnnData) -> ad.AnnData:
        x = normalize_log1p_if_needed(to_dense_float32(ref_eval), auto_log1p=self.auto_log1p)
        return _build_output_adata(ref_eval, x)


class PseudobulkShiftBaseline(BaseEmbryoBaseline):
    """Variance-preserving gene-wise mean & dispersion shift in log1p space.

    Unlike naive mean prediction (which collapses single-cell variance and destroys
    `mmd_u`, `energy_distance`, and `variogram`), this model estimates an L2-shrunk
    per-gene shift + gentle variance scaling on the training pair and transports every
    individual reference cell while preserving cell-to-cell heterogeneity and support
    sparsity.
    """

    name = "mean_shift"
    description = "Variance-preserving L2-shrunk pseudobulk shift (preserves single-cell heterogeneity)."

    def __init__(
        self,
        *,
        shrinkage: float = 0.85,
        variance_blend: float = 0.5,
        seed: int = 0,
        auto_log1p: bool = False,
    ) -> None:
        super().__init__(seed=seed, auto_log1p=auto_log1p)
        self.shrinkage = shrinkage
        self.variance_blend = variance_blend
        self.ref_mean_: np.ndarray | None = None
        self.delta_mean_: np.ndarray | None = None
        self.std_ratio_: np.ndarray | None = None
        self.target_q99_: np.ndarray | None = None
        self.celltype_deltas_: dict[str, np.ndarray] = {}

    def fit(self, ref_train: ad.AnnData, target_train: ad.AnnData) -> "PseudobulkShiftBaseline":
        x_ref, x_tgt = _align_expression(ref_train, target_train, self.auto_log1p)
        mu_ref = x_ref.mean(axis=0)
        mu_tgt = x_tgt.mean(axis=0)
        std_ref = x_ref.std(axis=0)
        std_tgt = x_tgt.std(axis=0)

        self.ref_mean_ = mu_ref
        self.delta_mean_ = self.shrinkage * (mu_tgt - mu_ref)
        raw_ratio = np.where(std_ref > 1e-4, std_tgt / np.maximum(std_ref, 1e-4), 1.0)
        blended = 1.0 + self.variance_blend * (np.clip(raw_ratio, 0.5, 2.0) - 1.0)
        self.std_ratio_ = blended.astype(np.float32)
        self.target_q99_ = np.quantile(x_tgt, 0.995, axis=0).astype(np.float32)

        # Learn optional celltype-stratified residual shifts when labels overlap cleanly
        self.celltype_deltas_ = {}
        if "celltype" in ref_train.obs.columns and "celltype" in target_train.obs.columns:
            ref_ct = ref_train.obs["celltype"].astype(str).to_numpy()
            tgt_ct = target_train.obs["celltype"].astype(str).to_numpy()
            shared = sorted(set(ref_ct) & set(tgt_ct))
            for ct in shared:
                r_mask = ref_ct == ct
                t_mask = tgt_ct == ct
                if int(r_mask.sum()) >= 5 and int(t_mask.sum()) >= 5:
                    n_eff = min(int(r_mask.sum()), int(t_mask.sum()))
                    weight = n_eff / (n_eff + 10.0)
                    ct_delta = x_tgt[t_mask].mean(axis=0) - x_ref[r_mask].mean(axis=0)
                    self.celltype_deltas_[ct] = (
                        weight * ct_delta + (1.0 - weight) * self.delta_mean_
                    ).astype(np.float32)
        return self

    def predict(self, ref_eval: ad.AnnData) -> ad.AnnData:
        if self.ref_mean_ is None or self.delta_mean_ is None or self.std_ratio_ is None:
            raise RuntimeError("Call fit() before predict().")
        x = normalize_log1p_if_needed(to_dense_float32(ref_eval), auto_log1p=self.auto_log1p)
        centered = (x - self.ref_mean_) * self.std_ratio_
        pred = self.ref_mean_ + centered + self.delta_mean_

        if self.celltype_deltas_ and "celltype" in ref_eval.obs.columns:
            ct_arr = ref_eval.obs["celltype"].astype(str).to_numpy()
            for ct, ct_delta in self.celltype_deltas_.items():
                mask = ct_arr == ct
                if np.any(mask):
                    pred[mask] = self.ref_mean_ + centered[mask] + ct_delta

        # Preserve physical support bounds without collapsing zeros into negative values
        if self.target_q99_ is not None:
            upper = np.maximum(self.target_q99_ * 1.15, x.max(axis=0))
            pred = np.minimum(pred, upper)
        pred = np.maximum(pred, 0.0)
        return _build_output_adata(ref_eval, pred)


class LatentAffineTransportBaseline(BaseEmbryoBaseline):
    """PCA/SVD latent Gaussian optimal-transport map (Monge map in top-k PCA space).

    Aligns both the first moment (mean shift -> DE recovery & direction) and second
    moment (covariance rotation/scaling -> cell-state distribution & gene-gene
    variogram) via closed-form Bures-Wasserstein linear transport in truncated SVD
    space, plus residual non-negative gene-space mean correction.
    """

    name = "latent_ot"
    description = "Closed-form Bures-Wasserstein (Gaussian OT) transport in truncated SVD space."

    def __init__(
        self,
        *,
        n_components: int = 24,
        cov_reg: float = 1e-2,
        seed: int = 0,
        auto_log1p: bool = False,
    ) -> None:
        super().__init__(seed=seed, auto_log1p=auto_log1p)
        self.n_components = n_components
        self.cov_reg = cov_reg
        self.svd_: TruncatedSVD | None = None
        self.mu_ref_g_: np.ndarray | None = None
        self.mu_tgt_g_: np.ndarray | None = None
        self.z_ref_mean_: np.ndarray | None = None
        self.z_tgt_mean_: np.ndarray | None = None
        self.transport_mat_: np.ndarray | None = None

    @staticmethod
    def _spd_sqrt_and_inv_sqrt(cov: np.ndarray, reg: float) -> tuple[np.ndarray, np.ndarray]:
        sym = 0.5 * (cov + cov.T) + reg * np.eye(cov.shape[0], dtype=np.float64)
        vals, vecs = np.linalg.eigh(sym)
        vals = np.maximum(vals, reg)
        sqrt_m = (vecs * np.sqrt(vals)) @ vecs.T
        inv_sqrt_m = (vecs * (1.0 / np.sqrt(vals))) @ vecs.T
        return sqrt_m, inv_sqrt_m

    def fit(self, ref_train: ad.AnnData, target_train: ad.AnnData) -> "LatentAffineTransportBaseline":
        x_ref, x_tgt = _align_expression(ref_train, target_train, self.auto_log1p)
        self.mu_ref_g_ = x_ref.mean(axis=0)
        self.mu_tgt_g_ = x_tgt.mean(axis=0)

        joint_centered = np.vstack([x_ref - self.mu_ref_g_, x_tgt - self.mu_tgt_g_])
        max_k = max(2, min(self.n_components, joint_centered.shape[0] - 1, joint_centered.shape[1] - 1))
        self.svd_ = TruncatedSVD(n_components=max_k, random_state=self.seed)
        self.svd_.fit(joint_centered)

        z_ref = self.svd_.transform(x_ref - self.mu_ref_g_).astype(np.float64)
        z_tgt = self.svd_.transform(x_tgt - self.mu_tgt_g_).astype(np.float64)
        self.z_ref_mean_ = z_ref.mean(axis=0)
        self.z_tgt_mean_ = z_tgt.mean(axis=0)

        c_ref = np.cov(z_ref, rowvar=False) if z_ref.shape[0] > 2 else np.eye(max_k)
        c_tgt = np.cov(z_tgt, rowvar=False) if z_tgt.shape[0] > 2 else np.eye(max_k)
        c_ref = np.atleast_2d(c_ref)
        c_tgt = np.atleast_2d(c_tgt)

        # Closed-form Gaussian OT map A = C_ref^{-1/2} (C_ref^{1/2} C_tgt C_ref^{1/2})^{1/2} C_ref^{-1/2}
        s_ref, inv_s_ref = self._spd_sqrt_and_inv_sqrt(c_ref, self.cov_reg)
        middle, _ = self._spd_sqrt_and_inv_sqrt(s_ref @ c_tgt @ s_ref, self.cov_reg * 1e-2)
        self.transport_mat_ = (inv_s_ref @ middle @ inv_s_ref).astype(np.float64)
        return self

    def predict(self, ref_eval: ad.AnnData) -> ad.AnnData:
        if (
            self.svd_ is None
            or self.mu_ref_g_ is None
            or self.mu_tgt_g_ is None
            or self.z_ref_mean_ is None
            or self.z_tgt_mean_ is None
            or self.transport_mat_ is None
        ):
            raise RuntimeError("Call fit() before predict().")

        x = normalize_log1p_if_needed(to_dense_float32(ref_eval), auto_log1p=self.auto_log1p)
        centered = x - self.mu_ref_g_
        z = self.svd_.transform(centered).astype(np.float64)
        # Preserve high-frequency orthogonal component so per-cell gene sparsity & variance survive
        x_low_rank = self.svd_.inverse_transform(z).astype(np.float32)
        orth_residual = centered - x_low_rank

        z_trans = (z - self.z_ref_mean_) @ self.transport_mat_.T + self.z_tgt_mean_
        x_trans_low_rank = self.svd_.inverse_transform(z_trans).astype(np.float32)

        pred = self.mu_tgt_g_ + x_trans_low_rank + 0.75 * orth_residual
        pred = np.maximum(pred, 0.0)
        return _build_output_adata(ref_eval, pred)


class SpatialGrowthTransportBaseline(BaseEmbryoBaseline):
    """Joint 3D spatial affine growth (Gaussian OT in R^3) + latent expression transport.

    Designed for T2 (3D MERFISH heart development) and T3 (3D KO perturbation):
      - Models 3D organ growth/morphogenesis via closed-form 3D covariance transport
        on `obsm['spatial_3D']` (improving `d2_shape`, `sliced_wasserstein`,
        `occupancy_dice`, and `scale_log_ratio`), AND
      - Applies `LatentAffineTransportBaseline` to gene expression so both transcriptomic
        and spatial-neighborhood (`neighborhood_mmd`) metrics improve simultaneously.
    Falls back cleanly to `LatentAffineTransportBaseline` if `spatial_3D` is absent (T1).
    """

    name = "spatial_growth_ot"
    description = "Joint 3D spatial Gaussian-OT growth map + latent SVD expression transport (T2/T3)."

    def __init__(
        self,
        *,
        n_components: int = 24,
        cov_reg: float = 1e-2,
        spatial_reg: float = 1e-3,
        seed: int = 0,
        auto_log1p: bool = False,
    ) -> None:
        super().__init__(seed=seed, auto_log1p=auto_log1p)
        self.spatial_reg = spatial_reg
        self.expr_model = LatentAffineTransportBaseline(
            n_components=n_components,
            cov_reg=cov_reg,
            seed=seed,
            auto_log1p=auto_log1p,
        )
        self.s_ref_mean_: np.ndarray | None = None
        self.s_tgt_mean_: np.ndarray | None = None
        self.s_transport_: np.ndarray | None = None

    def fit(self, ref_train: ad.AnnData, target_train: ad.AnnData) -> "SpatialGrowthTransportBaseline":
        self.expr_model.fit(ref_train, target_train)
        if "spatial_3D" in ref_train.obsm and "spatial_3D" in target_train.obsm:
            s_ref = np.asarray(ref_train.obsm["spatial_3D"], dtype=np.float64)
            s_tgt = np.asarray(target_train.obsm["spatial_3D"], dtype=np.float64)
            self.s_ref_mean_ = s_ref.mean(axis=0)
            self.s_tgt_mean_ = s_tgt.mean(axis=0)
            c_ref = np.atleast_2d(np.cov(s_ref, rowvar=False))
            c_tgt = np.atleast_2d(np.cov(s_tgt, rowvar=False))
            sq_ref, inv_sq_ref = LatentAffineTransportBaseline._spd_sqrt_and_inv_sqrt(
                c_ref, self.spatial_reg
            )
            mid, _ = LatentAffineTransportBaseline._spd_sqrt_and_inv_sqrt(
                sq_ref @ c_tgt @ sq_ref, self.spatial_reg * 1e-2
            )
            self.s_transport_ = inv_sq_ref @ mid @ inv_sq_ref
        else:
            self.s_ref_mean_ = None
            self.s_tgt_mean_ = None
            self.s_transport_ = None
        return self

    def predict(self, ref_eval: ad.AnnData) -> ad.AnnData:
        pred_ad = self.expr_model.predict(ref_eval)
        if (
            self.s_transport_ is not None
            and self.s_ref_mean_ is not None
            and self.s_tgt_mean_ is not None
            and "spatial_3D" in ref_eval.obsm
        ):
            s_eval = np.asarray(ref_eval.obsm["spatial_3D"], dtype=np.float64)
            s_pred = (s_eval - self.s_ref_mean_) @ self.s_transport_.T + self.s_tgt_mean_
            pred_ad.obsm["spatial_3D"] = s_pred.astype(np.float32)
        return pred_ad


BASELINE_REGISTRY: dict[str, type[BaseEmbryoBaseline]] = {
    "copy_last": CopyLastBaseline,
    "mean_shift": PseudobulkShiftBaseline,
    "latent_ot": LatentAffineTransportBaseline,
    "spatial_growth_ot": SpatialGrowthTransportBaseline,
}


def make_baseline(name: str, *, seed: int = 0, auto_log1p: bool = False) -> BaseEmbryoBaseline:
    if name not in BASELINE_REGISTRY:
        raise ValueError(f"Unknown baseline {name!r}. Choose from: {', '.join(BASELINE_REGISTRY)}")
    return BASELINE_REGISTRY[name](seed=seed, auto_log1p=auto_log1p)
