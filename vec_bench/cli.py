"""Command-line interface for vec-baseline-bench-cn."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad

from .baselines import BASELINE_REGISTRY, make_baseline
from .benchmark import render_markdown_table, run_benchmark
from .report_html import render_html_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vec-bench",
        description="Virtual Embryo Challenge (NeurIPS 2026) CPU-only starter baselines, "
                    "leakage-free local benchmark harness, and offline 3D HTML report generator.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Benchmark multiple CPU baselines locally with strict train/eval separation.")
    p_run.add_argument("--task", required=True, choices=["T1", "T2", "T3"])
    p_run.add_argument("--ref", required=True, type=Path, help="Reference stage (T1/T2) or WT (T3) .h5ad")
    p_run.add_argument("--target", required=True, type=Path, help="Pseudo-target stage or KO .h5ad")
    p_run.add_argument("--train-ref", type=Path, help="Optional separate training reference embryo .h5ad")
    p_run.add_argument("--train-target", type=Path, help="Optional separate training target embryo .h5ad")
    p_run.add_argument("--ceiling-target", type=Path, help="Optional biological replicate .h5ad for replicate ceiling")
    p_run.add_argument(
        "--baselines",
        help=f"Comma-separated list from: {', '.join(BASELINE_REGISTRY)} (default: task-appropriate set)",
    )
    p_run.add_argument("--train-frac", type=float, default=0.5, help="Disjoint train fraction when no --train-* files are passed")
    p_run.add_argument("--seed", type=int, default=0)
    p_run.add_argument("--auto-log1p", action="store_true", help="Automatically library-size + log1p normalize raw count matrices")
    p_run.add_argument("--setting", default="heart", choices=["heart", "embryo"])
    p_run.add_argument("--veckit-dir", type=Path, help="Optional path to local cloned aristoteleo/veckit")
    p_run.add_argument("--out-dir", type=Path, help="Write report.md, report.json, and offline report.html here")

    p_pred = sub.add_parser("predict", help="Fit one starter baseline and write a submission-ready .h5ad file.")
    p_pred.add_argument("--baseline", required=True, choices=list(BASELINE_REGISTRY))
    p_pred.add_argument("--train-ref", required=True, type=Path)
    p_pred.add_argument("--train-target", required=True, type=Path)
    p_pred.add_argument("--eval-ref", required=True, type=Path)
    p_pred.add_argument("--out", required=True, type=Path)
    p_pred.add_argument("--seed", type=int, default=0)
    p_pred.add_argument("--auto-log1p", action="store_true")

    args = parser.parse_args(argv)

    if args.cmd == "run":
        bnames = [x.strip() for x in args.baselines.split(",") if x.strip()] if args.baselines else None
        report = run_benchmark(
            args.task,
            args.ref,
            args.target,
            train_ref_path=args.train_ref,
            train_target_path=args.train_target,
            ceiling_target_path=args.ceiling_target,
            baseline_names=bnames,
            seed=args.seed,
            train_frac=args.train_frac,
            auto_log1p=args.auto_log1p,
            setting=args.setting,
            veckit_dir=args.veckit_dir,
        )
        md = render_markdown_table(report)
        print(md)
        if args.out_dir is not None:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            (args.out_dir / "report.md").write_text(md, encoding="utf-8")
            (args.out_dir / "report.json").write_text(
                json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
            )
            (args.out_dir / "report.html").write_text(render_html_report(report), encoding="utf-8")
            print(f"Wrote Markdown, JSON, and offline HTML report to {args.out_dir}")
        return 0

    if args.cmd == "predict":
        model = make_baseline(args.baseline, seed=args.seed, auto_log1p=args.auto_log1p)
        model.fit(ad.read_h5ad(args.train_ref), ad.read_h5ad(args.train_target))
        pred = model.predict(ad.read_h5ad(args.eval_ref))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        pred.write_h5ad(args.out)
        print(f"Wrote {args.baseline} prediction {pred.shape} to {args.out}")
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
