# Virtual Embryo Challenge（NeurIPS 2026）零 GPU 进阶基线与无泄漏本地评测中文指南

本文面向刚跑通官方 [`aristoteleo/veckit`](https://github.com/aristoteleo/veckit) 和 [`vec-preflight-cn`](https://github.com/XinOrigin/vec-preflight-cn) 的中文参赛者，重点回答三个实战问题：

1. **为什么直接预测均值细胞（Mean Collapse）会让 `mmd_u` 和 `variogram` 惨败？**
2. **在没有 A100/H100 的笔记本 CPU 上，如何写出比 `copy_last`（50 分基线）高 6–15 分的透明基线？**
3. **如何设计零数据泄漏的本地交叉验证（CV），避免在仅有 2 次机会的 P3 终审榜单上踩坑？**

---

## 一、为什么新手的第一个模型常常打不过 `copy_last`（50 分）？

官方榜单的四大问题组并不是单一的“基因均值误差（Pseudobulk MSE）”：
- **差异表达（`de_score` / `de_direction`）**：看基因层面的平均变化方向与显著性排序；
- **单细胞分布（`mmd_u` / `energy_distance`）**：看预测出的整个单细胞点云分布与真实目标点云的距离；
- **基因协变结构（`variogram`）**：看基因与基因之间在单细胞层面的相关矩阵是否保持真实结构；
- **3D 空间形态（T2/T3：`d2_shape` / `sliced_wasserstein` / `occupancy_dice` / `scale_log_ratio` / `neighborhood_mmd`）**：看胚胎器官在三维空间里的膨胀、形变与局部微环境。

如果你用一个回归模型只输出“目标阶段每个细胞类型的平均表达向量”，虽然 `DE` 看起来不错，但所有预测细胞坍缩到少数几个点，导致方差趋近 0——`mmd_u`、`energy_distance` 与 `variogram` 会瞬间崩盘，总分甚至低于原地不动的 `copy_last`（50 分）。

---

## 二、本工具库内置的 3 种零 GPU 进阶基线原理

### 1. `mean_shift`（保方差 L2 收缩伪批量平移 + 细胞类型分层）
- **做法**：在训练集 `(ref_train, target_train)` 上估计每个基因的收缩均值偏移 $\Delta \mu = \alpha (\mu_{\text{tgt}} - \mu_{\text{ref}})$ 与温和方法尺度比 $\sigma_{\text{tgt}} / \sigma_{\text{ref}}$；对评估集中的**每一个单细胞个体**做中心化尺度调整并叠加偏移（如果细胞类型标签重合，则按样本量加权融合该细胞类型的专属偏移），最后在 `log1p` 空间做非负截断。
- **特点**：完整保留单细胞异质性与稀疏支撑；在 32,285 基因的 T1 样本上仅需 **57 ms CPU**，即可把 `Headline Skill` 从 `50.00` 提升至 **`65.77`（+15.77）**。

### 2. `latent_ot`（截断 SVD 潜空间闭式 Bures-Wasserstein 最优传输）
- **做法**：在联合中心化表达矩阵上提取前 $k$ 维主成分（TruncatedSVD），在潜空间里直接求解高斯分布间 2-Wasserstein 最优传输（Monge 映射）闭式解：
  $$A = C_{\text{ref}}^{-1/2}\left(C_{\text{ref}}^{1/2} C_{\text{tgt}} C_{\text{ref}}^{1/2}\right)^{1/2} C_{\text{ref}}^{-1/2}$$
  同时保留潜空间正交残差项（`0.75 * orth_residual`），避免低秩重构抹掉高频单细胞稀疏特征。
- **特点**：同时对齐一阶均值与二阶协方差矩阵，在 T2 样本上拿到 **`63.92`（+13.92）** 且在 T1/T2/T3 上持续获得更优的 `mmd_u`。

### 3. `spatial_growth_ot`（3D 空间形态 Bures-Wasserstein 生长映射 + 潜空间表达传输）
- **做法**：专门针对 T2（3D MERFISH 心脏发育）和 T3（3D 基因敲除扰动），在 `obsm["spatial_3D"]`（$\mathbb{R}^3$）上同步拟合三维各向异性生长/形变最优传输映射，将细胞三维坐标一并演化到目标发育阶段或突变体形态。
- **消融实验铁证（T3 `sample_wt` → `sample_mab21l2_ko`）**：
  - 仅变换表达的 `latent_ot`：`46.44`
  - 加上 3D 空间生长映射的 `spatial_growth_ot`：**`56.97`（单靠 3D 坐标变换直接拉升 +10.53 分，仅耗时 4.0 ms CPU！）**

---

## 三、如何用本工具做“零泄漏”本地评测与离线可视化

### 1. 千万别在同一个数据集上既 `fit` 又 `score`
很多本地脚本容易在同一个 `.h5ad` 上既拟合参数又调用 `veckit` 打分，这会导致本地虚高、线上 P3 暴跌。`vec-bench run` 强制两种无泄漏模式：
- **模式 A（已有完整多胚胎训练集，最推荐）**：用 `--train-ref` / `--train-target` 传入胚胎 A，用 `--ref` / `--target` 传入未见胚胎 B，并用 `--ceiling-target` 传入胚胎 C 作为真实生物学重复上限。
- **模式 B（单对样本快速调试）**：自动对细胞索引做确定性 **50/50 严格互斥切分**（`fit` 与 `predict`/`score` 零细胞重叠）。

### 2. 一键生成免 CDN 离线 3D 空间对比报告
国内网络打开依赖外网 CDN 的 Plotly/JS 报告经常白屏。`vec-bench run --out-dir ./report_t3` 会生成纯本地自包含的 `report.html`，内置 **XY / XZ / YZ 三个正交平面的矢量 SVG 胚胎三维点云对比图**，双击即可在浏览器秒开，直观核查预测形态是否覆盖了目标器官轮廓。

### 3. 推荐的完整参赛工作流（与 `vec-preflight-cn` 联动）
1. 用 `vec-bench run` 跑通无泄漏本地消融实验，选出在目标任务上稳定超越 `50.0` 的方法；
2. 用 `vec-bench predict`（或你基于 `BaseEmbryoBaseline` 扩展的新模型）生成候选 `.h5ad`；
3. 用 [`vec-preflight-cn`](https://github.com/XinOrigin/vec-preflight-cn) 执行 `vec-preflight check` 排雷（检查非负、无 NaN、`log1p` 尺度、`obsm['spatial_3D']` 键名与基因顺序）；
4. 确认全绿后再使用宝贵的 P3 正式提交名额。
