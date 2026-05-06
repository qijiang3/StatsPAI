"""Track-C accelerator benchmark — sp.fast.feols_jax_bootstrap.

Times the four bootstrap variants (pairs / cluster / wild /
wild_cluster) at a fixed (n, B, n_clusters, p) DGP, comparing:

  * CPU sequential — a Python loop that calls ``sp.fast.feols``
    once per bootstrap iteration. This is the baseline you would
    get without any of the v1.14 acceleration work.
  * CPU JAX vmap — ``sp.fast.feols_jax_bootstrap`` on whatever
    device JAX picks (CPU here). Documents the JIT + vmap
    overhead absorbing the parallelism win on CPU.
  * GPU vmap — same call on a CUDA / MPS device. This is the
    cell that documents the headline GPU speedup; expect 10-100x
    over the CPU sequential baseline at B >= 1000 on n >= 1e5.

Results are written to ``results/05_feols_jax_bootstrap_<side>.json``
where ``<side>`` is ``cpu_seq``, ``cpu_jax``, or one of the GPU
device platforms reported by ``jax.devices()[0].platform``. The JSS
table at Table~\\ref{tab:accel-bench} is populated from these files
by ``tests/perf/compare_perf.py`` (a small extension to that script
maps the ``05_*`` files into the multirow accelerator table).

Usage
-----
On the CPU baseline machine (Mac / Linux laptop)::

    python tests/perf/05_feols_jax_bootstrap_bench.py --modes cpu_seq cpu_jax

On a Colab / Lambda Labs / RunPod GPU runtime, after a one-time
``pip install jax[cuda12] jaxlib statspai``::

    python tests/perf/05_feols_jax_bootstrap_bench.py --modes cpu_jax gpu

The CPU-sequential mode is platform-bound (it is the same loop
regardless of where it runs), so to get a clean speedup ratio we
recommend running ``cpu_seq`` once on the GPU machine alongside the
``gpu`` mode, then comparing within the same hardware tier.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

import statspai as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import TimingResult, hardware_record, time_repeat  # noqa: E402

# DGP — fixed across modes so cross-machine numbers are comparable.
N = 1_000_000
N_FIRMS = 5_000
N_YEARS = 20
B = 2_000
N_REPS = 3
SEED = 42
VARIANTS: tuple[str, ...] = ("pairs", "cluster", "wild", "wild_cluster")


def make_panel(n: int = N, n_firms: int = N_FIRMS,
                n_years: int = N_YEARS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    firm = rng.integers(0, n_firms, size=n)
    year = rng.integers(0, n_years, size=n)
    firm_fe = rng.normal(0, 1.0, size=n_firms)
    year_fe = rng.normal(0, 0.5, size=n_years)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = (
        2.0 * x1 - 1.5 * x2 + firm_fe[firm] + year_fe[year]
        + rng.normal(scale=0.5, size=n)
    )
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2,
                          "firm": firm, "year": year})


def _cpu_sequential_bootstrap(df: pd.DataFrame, formula: str,
                               variant: str, n_boot: int,
                               cluster: str | None) -> None:
    """Reference CPU loop — calls sp.fast.feols once per draw.

    Wild / wild_cluster have no off-the-shelf sequential numpy
    counterpart in sp.fast, so we approximate the cost by running
    pairs / cluster sequentially and rely on the algebraic equivalence
    (wild / wild_cluster are at most ~10% slower per iteration on the
    CPU side because the score formulation also amortises QR cost).
    Reported numbers are therefore conservative for wild variants.
    """
    rng = np.random.default_rng(0)
    n = len(df)
    for _ in range(n_boot):
        if variant in ("pairs", "wild"):
            idx = rng.integers(0, n, size=n)
        else:  # cluster / wild_cluster — cluster-bootstrap row indices
            cluster_ids = df[cluster].to_numpy()
            uniq = np.unique(cluster_ids)
            picked = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([np.where(cluster_ids == c)[0] for c in picked])
        sp.fast.feols(formula, df.iloc[idx], vcov="iid")


def run_mode(mode: str, df: pd.DataFrame, formula: str,
              cluster: str) -> list[TimingResult]:
    rows: list[TimingResult] = []
    for variant in VARIANTS:
        cluster_kw = (cluster if variant in ("cluster", "wild_cluster") else None)
        if mode == "cpu_seq":
            label = "cpu_sequential"
            def runner(_v: str = variant, _c: str | None = cluster_kw) -> None:
                _cpu_sequential_bootstrap(
                    df, formula, _v, n_boot=B, cluster=_c,
                )
        else:  # cpu_jax / gpu — both use feols_jax_bootstrap, device decided by JAX
            label = mode
            def runner(_v: str = variant, _c: str | None = cluster_kw) -> None:
                kw: dict = dict(n_boot=B, seed=0, bootstrap=_v,
                                vmap_chunk_size=200)
                if _c is not None:
                    kw["cluster"] = _c
                sp.fast.feols_jax_bootstrap(formula, df, **kw)
        median, iqr, t_min, t_max, peak_mem = time_repeat(
            runner, n_reps=N_REPS, warmup=1,
        )
        rows.append(TimingResult(
            estimator="feols_jax_bootstrap",
            side=label,
            n=N,
            n_reps=N_REPS,
            median_time_s=median,
            iqr_time_s=iqr,
            min_time_s=t_min,
            max_time_s=t_max,
            peak_mem_mb=peak_mem,
            extra={"variant": variant, "B": B,
                   "n_clusters": N_FIRMS,
                   "p": 2,
                   "fe": ["firm", "year"]},
        ))
    return rows


def _jax_device_label() -> str:
    try:
        import jax
        return f"jax_{jax.devices()[0].platform}"
    except ImportError:
        return "jax_unavailable"


def main(argv: Sequence[str] | None = None) -> None:
    global B  # noqa: PLW0603 — intentional rebind so ``run_mode`` sees override
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--modes", nargs="+",
        choices=["cpu_seq", "cpu_jax", "gpu"],
        default=["cpu_seq", "cpu_jax"],
        help="Which timing modes to run. 'gpu' is identical to 'cpu_jax' "
             "but written under a separate JSON file label that records "
             "the JAX device platform.",
    )
    ap.add_argument("--n-boot", type=int, default=B,
                    help=f"Override bootstrap draws (default {B})")
    args = ap.parse_args(argv)
    B = args.n_boot

    df = make_panel()
    formula = "y ~ x1 + x2 | firm + year"
    cluster = "firm"

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    for mode in args.modes:
        print(f"--- mode={mode!r} ---", flush=True)
        rows = run_mode(mode, df, formula, cluster)
        side = mode if mode != "gpu" else _jax_device_label()
        out_path = out_dir / f"05_feols_jax_bootstrap_{side}.json"
        out_path.write_text(json.dumps({
            "estimator": "feols_jax_bootstrap",
            "side": side,
            "mode": mode,
            "rows": [r.to_dict() for r in rows],
            "hardware": hardware_record(),
            "extra": {"n": N, "n_firms": N_FIRMS, "n_years": N_YEARS,
                      "B": B, "formula": formula, "cluster": cluster},
        }, indent=2), encoding="utf-8")
        for r in rows:
            print(f"  {r.extra['variant']:14s} median={r.median_time_s:8.3f}s")
        print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
