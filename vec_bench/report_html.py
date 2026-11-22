"""Self-contained, zero-CDN offline HTML report generator with inline 3D SVG projections."""
from __future__ import annotations

import html
from typing import Any

import numpy as np


def _render_2d_projection_svg(
    title: str,
    clouds: dict[str, list[list[float]]],
    ax_i: int,
    ax_j: int,
    labels: tuple[str, str],
    width: int = 320,
    height: int = 260,
) -> str:
    """Render an inline SVG orthogonal projection (XY, XZ, or YZ) of 3D embryo coordinates."""
    palette = {
        "reference": ("#94a3b8", "Ref (t)", 0.45),
        "target": ("#10b981", "Held-out Target", 0.72),
        "spatial_growth_ot": ("#3b82f6", "spatial_growth_ot", 0.72),
        "latent_ot": ("#8b5cf6", "latent_ot", 0.65),
        "mean_shift": ("#f59e0b", "mean_shift", 0.65),
    }
    ordered_keys = [k for k in ("reference", "target", "spatial_growth_ot", "latent_ot", "mean_shift") if k in clouds]
    if not ordered_keys:
        return ""

    all_pts = np.vstack([np.asarray(clouds[k], dtype=float)[:, [ax_i, ax_j]] for k in ordered_keys])
    xmin, ymin = all_pts.min(axis=0)
    xmax, ymax = all_pts.max(axis=0)
    dx = max(xmax - xmin, 1e-6)
    dy = max(ymax - ymin, 1e-6)
    pad = 28

    def map_x(v: float) -> float:
        return pad + (v - xmin) / dx * (width - 2 * pad)

    def map_y(v: float) -> float:
        return height - pad - (v - ymin) / dy * (height - 2 * pad)

    circles: list[str] = []
    # Cap drawn points per cloud for snappy offline HTML size
    for key in ordered_keys:
        pts = np.asarray(clouds[key], dtype=float)
        step = max(1, len(pts) // 250)
        color, _, alpha = palette.get(key, ("#64748b", key, 0.6))
        for p in pts[::step]:
            circles.append(
                f'<circle cx="{map_x(p[ax_i]):.1f}" cy="{map_y(p[ax_j]):.1f}" '
                f'r="2.6" fill="{color}" fill-opacity="{alpha}"/>'
            )

    legend_items = []
    lx = pad
    for key in ordered_keys:
        color, lbl, _ = palette.get(key, ("#64748b", key, 0.6))
        legend_items.append(
            f'<circle cx="{lx}" cy="14" r="4" fill="{color}"/>'
            f'<text x="{lx + 8}" y="18" font-size="10" fill="#cbd5e1">{html.escape(lbl)}</text>'
        )
        lx += len(lbl) * 6 + 24

    return f"""
    <div class="svg-card">
      <div class="svg-title">{html.escape(title)}</div>
      <svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
        <rect x="0" y="0" width="{width}" height="{height}" rx="8" fill="#0f172a"/>
        <rect x="{pad}" y="{pad}" width="{width - 2 * pad}" height="{height - 2 * pad}" fill="none" stroke="#334155" stroke-width="1"/>
        {''.join(legend_items)}
        {''.join(circles)}
        <text x="{width / 2:.1f}" y="{height - 6}" text-anchor="middle" font-size="10" fill="#94a3b8">{html.escape(labels[0])}</text>
        <text x="10" y="{height / 2:.1f}" text-anchor="middle" font-size="10" fill="#94a3b8" transform="rotate(-90 10 {height / 2:.1f})">{html.escape(labels[1])}</text>
      </svg>
    </div>
    """


def render_html_report(report: dict[str, Any]) -> str:
    """Build a standalone, zero-CDN interactive HTML benchmark report."""
    rows_html: list[str] = []
    for idx, row in enumerate(report["results"], 1):
        delta = row["delta_vs_floor"]
        badge_cls = "gain" if delta > 0.05 else ("flat" if abs(delta) <= 0.05 else "drop")
        group_pills = " ".join(
            f'<span class="pill">{html.escape(k)}: {"N/A" if v is None else f"{v:.1f}"}</span>'
            for k, v in row["group_scores_100"].items()
        )
        rm = row["raw_metrics"]
        de = f"{rm['de_score']:.4f}" if isinstance(rm.get("de_score"), (int, float)) else "—"
        ed = f"{rm['energy_distance']:.4f}" if isinstance(rm.get("energy_distance"), (int, float)) else "—"
        mmd = f"{rm['mmd_u']:.4f}" if isinstance(rm.get("mmd_u"), (int, float)) else "—"
        sw = f"{rm['sliced_wasserstein']:.4f}" if isinstance(rm.get("sliced_wasserstein"), (int, float)) else "—"
        rows_html.append(
            f"""
            <tr>
              <td>#{idx}</td>
              <td><code>{html.escape(row['baseline'])}</code><div class="sub">{html.escape(row['description'])}</div></td>
              <td class="num"><strong>{row['headline_skill_100']:.2f}</strong></td>
              <td class="num"><span class="badge {badge_cls}">{delta:+.2f}</span></td>
              <td class="num">{de}</td>
              <td class="num">{ed}</td>
              <td class="num">{mmd}</td>
              <td class="num">{sw}</td>
              <td>{group_pills}</td>
              <td class="num">{row['elapsed_ms']:.1f} ms</td>
            </tr>
            """
        )

    clouds = report.get("spatial_clouds") or {}
    spatial_section = ""
    if clouds:
        # Compare reference, held-out target, and the top spatial model
        subset = {k: clouds[k] for k in ("reference", "target", "spatial_growth_ot") if k in clouds}
        if len(subset) < 3 and "mean_shift" in clouds:
            subset["mean_shift"] = clouds["mean_shift"]
        svgs = "".join([
            _render_2d_projection_svg("XY Projection (3D Spatial)", subset, 0, 1, ("X", "Y")),
            _render_2d_projection_svg("XZ Projection (3D Spatial)", subset, 0, 2, ("X", "Z")),
            _render_2d_projection_svg("YZ Projection (3D Spatial)", subset, 1, 2, ("Y", "Z")),
        ])
        spatial_section = f"""
        <section>
          <h2>3D Embryo Spatial Projections (Offline SVG) / 3D 胚胎空间投影对比</h2>
          <p class="sub">Comparing Reference stage, Held-out Target, and predicted <code>obsm['spatial_3D']</code> point clouds across XY, XZ, and YZ planes.</p>
          <div class="svg-grid">{svgs}</div>
        </section>
        """

    inv = report["results"][0].get("inverted_anchors", []) if report["results"] else []
    warn_html = (
        f'<div class="warn"><strong>Anchor diagnostic:</strong> Small-subsample split-half ceiling was inverted for '
        f'<code>{html.escape(", ".join(inv))}</code> and excluded from rescaled headline skill. '
        f'Evaluate on full-size embryos (&ge;1,000 cells) or pass <code>--ceiling-target</code>.</div>'
        if inv else ""
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>VEC Baseline Benchmark — {html.escape(report['task'])}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #020617; color: #e2e8f0; margin: 0; padding: 28px; line-height: 1.5; }}
  .wrap {{ max-width: 1120px; margin: 0 auto; }}
  h1 {{ font-size: 24px; margin: 0 0 8px; color: #f8fafc; }}
  h2 {{ font-size: 18px; margin: 28px 0 10px; color: #cbd5e1; }}
  .meta {{ color: #94a3b8; font-size: 13px; margin-bottom: 18px; }}
  .sub {{ color: #94a3b8; font-size: 12px; }}
  code {{ background: #1e293b; padding: 2px 6px; border-radius: 4px; color: #93c5fd; font-size: 12px; }}
  table {{ width: 100%; border-collapse: collapse; background: #0f172a; border-radius: 10px; overflow: hidden; font-size: 13px; }}
  th, td {{ padding: 10px 12px; border-bottom: 1px solid #1e293b; text-align: left; vertical-align: top; }}
  th {{ background: #1e293b; color: #cbd5e1; font-weight: 600; }}
  td.num, th.num {{ text-align: right; white-space: nowrap; }}
  .badge {{ padding: 2px 8px; border-radius: 999px; font-weight: 600; font-size: 12px; }}
  .gain {{ background: rgba(16,185,129,0.2); color: #34d399; }}
  .flat {{ background: rgba(148,163,184,0.2); color: #cbd5e1; }}
  .drop {{ background: rgba(239,68,68,0.2); color: #f87171; }}
  .pill {{ display: inline-block; background: #1e293b; color: #cbd5e1; border-radius: 4px; padding: 1px 6px; margin: 1px; font-size: 11px; }}
  .warn {{ margin-top: 14px; padding: 10px 14px; border-left: 4px solid #f59e0b; background: rgba(245,158,11,0.1); border-radius: 6px; font-size: 13px; }}
  .svg-grid {{ display: flex; flex-wrap: wrap; gap: 16px; margin-top: 10px; }}
  .svg-card {{ background: #0f172a; border: 1px solid #1e293b; border-radius: 10px; padding: 10px; }}
  .svg-title {{ font-size: 13px; font-weight: 600; margin-bottom: 6px; color: #cbd5e1; }}
  footer {{ margin-top: 28px; color: #64748b; font-size: 12px; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>Virtual Embryo Challenge ({html.escape(report['task'])}) — CPU Baseline Benchmark</h1>
  <div class="meta">
    Split: <code>{html.escape(report['split_mode'])}</code> &middot;
    Ceiling: <code>{html.escape(report['ceiling_mode'])}</code> &middot;
    auto_log1p: <code>{report['auto_log1p']}</code> &middot;
    Eval ref: <code>{report['shapes']['ref_eval']}</code> &middot;
    Eval target: <code>{report['shapes']['tgt_eval']}</code>
  </div>
  <table>
    <thead>
      <tr>
        <th>Rank</th><th>Baseline</th><th class="num">Skill (0–100)</th><th class="num">vs Floor</th>
        <th class="num">DE</th><th class="num">Energy</th><th class="num">MMD</th><th class="num">SW-3D</th>
        <th>Question Groups</th><th class="num">CPU Time</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows_html)}
    </tbody>
  </table>
  {warn_html}
  {spatial_section}
  <footer>Generated offline by <code>vec-baseline-bench-cn</code> (zero external CDN dependencies). Local pseudo-target metrics verify pipeline correctness and relative gains over <code>copy_last</code>, not final scores on hidden held-out test sets.</footer>
</div>
</body>
</html>
"""
