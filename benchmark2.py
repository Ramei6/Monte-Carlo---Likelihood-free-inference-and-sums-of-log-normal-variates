"""
benchmark.py
============
Compare Reject-ABC vs MCMC-ABC across a shared ε grid.

Protocol
--------
* R independent repetitions; each rep draws a fresh y_obs (n = 1 000).
* Both algorithms see the SAME y_obs within every rep.
* JIT kernels are pre-compiled once before the timed loop.
* For each (rep, ε):
    - time Reject-ABC  →  record runtime, posterior means, acc rate, ESS
    - time MCMC-ABC    →  same
* Save raw records to CSV, then produce a 6-panel comparison figure.

Key design choices
------------------
* Both algorithms live in algorithms.py (clean alg1/alg2 interface).
* Reject-ABC: K=1 (single simulation per proposal).
* MCMC-ABC  : K=1 (single simulation per step), δ=0.2 (near-optimal from Q2).
* ESS for Reject-ABC = n_keep (i.i.d. draws — by construction).
* ESS for MCMC-ABC   = truncated-ACF estimate on the μ chain.
* ESS/second is the primary efficiency metric.

Outputs
-------
  benchmark_results/benchmark_results.csv
  benchmark_results/benchmark_plots.png
"""

import os
import time

import numpy as np
import jax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

import algorithms


os.makedirs("benchmark_results", exist_ok=True)


# ── Configuration ──────────────────────────────────────────────────────────────
N_OBS    = 1_000
R        = 3
EPS_GRID = np.array([0.5, 0.7, 1.0, 1.5, 2.0, 3.0])

# Reject-ABC settings
N_KEEP   = algorithms.N_KEEP_REJ

# MCMC-ABC settings
DELTA    = algorithms.DELTA_MCMC
N_BURN   = algorithms.N_BURN_MCMC
N_ITER   = algorithms.N_ITER_MCMC


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark loop
# ═══════════════════════════════════════════════════════════════════════════════

def run_benchmark() -> pd.DataFrame:
    records   = []
    key_bench = jax.random.PRNGKey(2026)

    print("=" * 72)
    print("  Benchmark: Reject-ABC  vs  MCMC-ABC  (K=1 for both)")
    print("=" * 72)
    print(f"  Model : Y_i = Σ exp(X_il),  L={algorithms.L},  n={N_OBS}")
    print(f"  Truth : μ₀={algorithms.TRUE_MU},  σ₀²={algorithms.TRUE_SIGMA2}")
    print(
        f"  Prior : μ~N(0,{algorithms.S_PRIOR}²),  "
        f"log σ²~N(0,{algorithms.T_PRIOR}²)"
    )
    print(f"  Reps  : R={R}")
    print(f"  ε grid: {EPS_GRID}")
    print(f"  Reject-ABC : n_keep={N_KEEP}  (ESS = n_keep, i.i.d. draws)")
    print(
        f"  MCMC-ABC   : n_burn={N_BURN}, n_iter={N_ITER}, δ={DELTA}  "
        f"(ESS via truncated ACF)\n"
    )

    # ── Pre-compile JAX kernels ──────────────────────────────────────────────
    print("Warming up JIT (one MCMC chain compile + run — ~5-15 s)...")
    t_wup = time.perf_counter()
    algorithms.warmup(n_obs=N_OBS, eps=float(EPS_GRID[2]))
    print(f"  Done in {time.perf_counter() - t_wup:.1f} s\n")

    # ── Main loop ────────────────────────────────────────────────────────────
    for r in range(R):
        key_bench, k_data = jax.random.split(key_bench)
        y_obs = np.asarray(
            algorithms.generate_dataset(k_data, n=N_OBS)
        )

        print(
            f"─── Rep {r + 1}/{R}  "
            f"(mean={y_obs.mean():.2f}, std={y_obs.std():.2f}) ───"
        )

        for i_eps, eps in enumerate(EPS_GRID):
            seed_base = r * 1000 + i_eps * 10

            # ── Reject-ABC ──────────────────────────────────────────────────
            try:
                t0 = time.perf_counter()
                mu_r, s2_r, rate_r, ess_r = algorithms.reject_abc(
                    y_obs,
                    float(eps),
                    n_keep=N_KEEP,
                    seed=seed_base,
                )
                rt_r = time.perf_counter() - t0
                mean_mu_r = float(mu_r.mean())
                mean_s2_r = float(s2_r.mean())

            except Exception as exc:
                print(f"    [Reject-ABC ε={eps:.1f}] ERROR: {exc}")
                rt_r = mean_mu_r = mean_s2_r = rate_r = ess_r = float("nan")

            records.append(dict(
                rep=r,
                epsilon=float(eps),
                method="Reject-ABC",
                runtime_s=rt_r,
                mean_mu=mean_mu_r,
                mean_sigma2=mean_s2_r,
                acc_rate=rate_r,
                ess=ess_r,
                ess_per_sec=ess_r / rt_r if rt_r > 0 else float("nan"),
            ))

            # ── MCMC-ABC ────────────────────────────────────────────────────
            try:
                t0 = time.perf_counter()
                mu_m, s2_m, rate_m, ess_m = algorithms.mcmc_abc(
                    y_obs,
                    float(eps),
                    delta=DELTA,
                    n_iter=N_ITER,
                    n_burn=N_BURN,
                    seed=seed_base + 1,
                )
                rt_m = time.perf_counter() - t0
                mean_mu_m = float(mu_m.mean())
                mean_s2_m = float(s2_m.mean())

            except Exception as exc:
                print(f"    [MCMC-ABC ε={eps:.1f}] ERROR: {exc}")
                rt_m = mean_mu_m = mean_s2_m = rate_m = ess_m = float("nan")

            records.append(dict(
                rep=r,
                epsilon=float(eps),
                method="MCMC-ABC",
                runtime_s=rt_m,
                mean_mu=mean_mu_m,
                mean_sigma2=mean_s2_m,
                acc_rate=rate_m,
                ess=ess_m,
                ess_per_sec=ess_m / rt_m if rt_m > 0 else float("nan"),
            ))

            rej_ess_per_sec = ess_r / rt_r if rt_r > 0 else float("nan")
            mcmc_ess_per_sec = ess_m / rt_m if rt_m > 0 else float("nan")

            print(
                f"  ε={eps:.1f} | "
                f"Rej : {rt_r:6.1f}s  acc={rate_r:.3f}  "
                f"ESS={ess_r:5.0f}  ESS/s={rej_ess_per_sec:5.1f} | "
                f"MCMC: {rt_m:6.1f}s  acc={rate_m:.3f}  "
                f"ESS={ess_m:5.0f}  ESS/s={mcmc_ess_per_sec:5.1f}"
            )

        print()

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════════
# Plots — 6-panel comparison figure
# ═══════════════════════════════════════════════════════════════════════════════

def make_plots(df: pd.DataFrame) -> None:
    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "font.size":         11,
        "axes.titlesize":    12,
        "axes.titleweight":  "bold",
        "axes.labelsize":    11,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "legend.fontsize":   9,
        "legend.framealpha": 0.9,
        "xtick.labelsize":   10,
        "ytick.labelsize":   10,
    })

    grp = df.groupby(["method", "epsilon"])
    mean = grp.mean(numeric_only=True).reset_index()
    std = grp.std(numeric_only=True).reset_index()

    def _get(method):
        m = (
            mean[mean.method == method]
            .sort_values("epsilon")
            .reset_index(drop=True)
        )
        s = (
            std[std.method == method]
            .sort_values("epsilon")
            .reset_index(drop=True)
        )
        return m, s

    Rm, Rs = _get("Reject-ABC")
    Mm, Ms = _get("MCMC-ABC")

    C_R = "#C0392B"
    C_M = "#2471A3"

    eps = Rm.epsilon.values

    # Bigger figure to avoid overlap
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(19, 11),
        constrained_layout=False,
    )

    fig.patch.set_facecolor("#F8F9FA")

    for ax in axes.flat:
        ax.set_facecolor("#FFFFFF")
        ax.set_xticks(eps)

    # Main title and subtitle
    fig.suptitle(
        "Reject-ABC  vs  MCMC-ABC  —  Benchmark comparison",
        fontsize=16,
        fontweight="bold",
        color="#2C3E50",
        y=0.975,
    )

    fig.text(
        0.5,
        0.945,
        r"$Y_i = \sum_\ell e^{X_{i\ell}}$,  "
        r"truth: $\mu_0=0,\ \sigma_0^2=0.09$,  $n=1000,\ L=10$"
        f"  |  mean ± 1 std over {R} reps",
        ha="center",
        fontsize=10,
        color="#555",
    )

    def _band(ax, x, col, m_df, s_df, color, marker, ls, label, lw=2.2, ms=7):
        y = m_df[col].values
        dy = s_df[col].fillna(0).values

        ax.plot(
            x,
            y,
            marker=marker,
            ls=ls,
            color=color,
            lw=lw,
            ms=ms,
            label=label,
            zorder=3,
        )

        ax.fill_between(
            x,
            y - dy,
            y + dy,
            color=color,
            alpha=0.12,
            zorder=2,
        )

    # ── (a) Runtime ───────────────────────────────────────────────────────────
    ax = axes[0, 0]

    _band(ax, eps, "runtime_s", Rm, Rs, C_R, "o", "-",  "Reject-ABC")
    _band(ax, eps, "runtime_s", Mm, Ms, C_M, "s", "--", "MCMC-ABC")

    ax.set_yscale("log")

    # Add padding so annotations do not touch the top border
    ymin = min(Rm["runtime_s"].min(), Mm["runtime_s"].min())
    ymax = max(Rm["runtime_s"].max(), Mm["runtime_s"].max())
    ax.set_ylim(ymin * 0.75, ymax * 1.45)

    ax.set_xlabel("Tolerance  ε")
    ax.set_ylabel("Wall-clock time (s)  [log scale]")
    ax.set_title("(a)  Runtime vs ε", pad=10)
    ax.legend(loc="upper right")
    ax.grid(True, which="both", alpha=0.2, linestyle="--")

    # Annotate runtime values
    for xi, yi, yj in zip(eps, Rm["runtime_s"].values, Mm["runtime_s"].values):
        ax.annotate(
            f"{yi:.1f}s",
            (xi, yi),
            textcoords="offset points",
            xytext=(0, 7),
            fontsize=7.5,
            color=C_R,
            ha="center",
        )

        ax.annotate(
            f"{yj:.1f}s",
            (xi, yj),
            textcoords="offset points",
            xytext=(0, -14),
            fontsize=7.5,
            color=C_M,
            ha="center",
        )

    # ── (b) Speedup ratio ─────────────────────────────────────────────────────
    ax = axes[0, 1]

    ratio = Rm["runtime_s"].values / Mm["runtime_s"].values
    colors_bar = [C_R if v > 1 else C_M for v in ratio]

    bars = ax.bar(
        eps,
        ratio,
        width=0.18,
        color=colors_bar,
        alpha=0.80,
        edgecolor="white",
        linewidth=1.2,
        zorder=3,
    )

    ax.axhline(
        1,
        color="#444",
        lw=1.6,
        ls="--",
        zorder=4,
        label="Equal speed  (ratio = 1)",
    )

    for bar, v in zip(bars, ratio):
        if v >= 1:
            ypos = v + 0.15
        else:
            ypos = v + 0.08

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            ypos,
            f"{v:.1f}×",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color="#222",
        )

    ax.fill_between(
        [eps[0] - 0.4, eps[-1] + 0.4],
        [0, 0],
        [1, 1],
        color=C_M,
        alpha=0.05,
    )

    ax.fill_between(
        [eps[0] - 0.4, eps[-1] + 0.4],
        [1, 1],
        [max(ratio) * 1.35] * 2,
        color=C_R,
        alpha=0.05,
    )

    ax.set_xlabel("Tolerance  ε")
    ax.set_ylabel("Reject runtime / MCMC runtime")
    ax.set_title("(b)  Speedup ratio  (Reject ÷ MCMC)", pad=10)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.2, linestyle="--")
    ax.set_ylim(0, max(ratio) * 1.45)

    # ── (c) ESS / second ──────────────────────────────────────────────────────
    ax = axes[0, 2]

    _band(ax, eps, "ess_per_sec", Rm, Rs, C_R, "o", "-",  "Reject-ABC")
    _band(ax, eps, "ess_per_sec", Mm, Ms, C_M, "s", "--", "MCMC-ABC")

    ax.set_xlabel("Tolerance  ε")
    ax.set_ylabel("ESS / second")
    ax.set_title("(c)  Efficiency: ESS / second  vs ε", pad=10)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.2, linestyle="--")

    ax.text(
        0.97,
        0.97,
        "Higher = more independent\nsamples per unit time",
        transform=ax.transAxes,
        fontsize=8,
        color="#555",
        ha="right",
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
    )

    # ── (d) Posterior mean μ ──────────────────────────────────────────────────
    ax = axes[1, 0]

    _band(ax, eps, "mean_mu", Rm, Rs, C_R, "o", "-",  "Reject-ABC")
    _band(ax, eps, "mean_mu", Mm, Ms, C_M, "s", "--", "MCMC-ABC")

    ax.axhline(
        algorithms.TRUE_MU,
        color="#2C3E50",
        lw=1.8,
        ls=":",
        label=f"Truth  μ₀ = {algorithms.TRUE_MU}",
    )

    ax.set_xlabel("Tolerance  ε")
    ax.set_ylabel("Posterior mean  E[μ | y*]")
    ax.set_title("(d)  Posterior mean  μ  vs ε", pad=10)
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.2, linestyle="--")

    # ── (e) Posterior mean σ² ─────────────────────────────────────────────────
    ax = axes[1, 1]

    _band(ax, eps, "mean_sigma2", Rm, Rs, C_R, "o", "-",  "Reject-ABC")
    _band(ax, eps, "mean_sigma2", Mm, Ms, C_M, "s", "--", "MCMC-ABC")

    ax.axhline(
        algorithms.TRUE_SIGMA2,
        color="#2C3E50",
        lw=1.8,
        ls=":",
        label=f"Truth  σ₀² = {algorithms.TRUE_SIGMA2:.2f}",
    )

    ax.set_xlabel("Tolerance  ε")
    ax.set_ylabel("Posterior mean  E[σ² | y*]")
    ax.set_title("(e)  Posterior mean  σ²  vs ε", pad=10)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.2, linestyle="--")

    # ── (f) Acceptance rate ───────────────────────────────────────────────────
    ax = axes[1, 2]

    _band(ax, eps, "acc_rate", Rm, Rs, C_R, "o", "-",  "Reject-ABC")
    _band(ax, eps, "acc_rate", Mm, Ms, C_M, "s", "--", "MCMC-ABC")

    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:.1%}")
    )

    ax.set_xlabel("Tolerance  ε")
    ax.set_ylabel("Acceptance rate")
    ax.set_title("(f)  Acceptance rate  vs ε", pad=10)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.2, linestyle="--")

    # Manual spacing. This is the key layout fix.
    fig.subplots_adjust(
        left=0.06,
        right=0.985,
        bottom=0.08,
        top=0.89,
        wspace=0.25,
        hspace=0.34,
    )

    path = "benchmark_results/benchmark2_plots.png"

    fig.savefig(
        path,
        dpi=180,
        facecolor="#F8F9FA",
    )

    plt.close(fig)
    plt.rcParams.update(plt.rcParamsDefault)

    print(f"Plots saved  →  {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    df = run_benchmark()

    csv_path = "benchmark_results/benchmark_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Results saved  →  {csv_path}\n")

    # Summary table
    agg = (
        df.groupby(["method", "epsilon"])
        .mean(numeric_only=True)[[
            "runtime_s",
            "acc_rate",
            "ess",
            "ess_per_sec",
            "mean_mu",
            "mean_sigma2",
        ]]
        .reset_index()
    )

    for col in ["mean_mu", "mean_sigma2"]:
        agg[col] = agg[col].map("{:.4f}".format)

    agg["acc_rate"] = agg["acc_rate"].map("{:.3f}".format)
    agg["runtime_s"] = agg["runtime_s"].map("{:.2f}s".format)
    agg["ess"] = agg["ess"].map("{:.0f}".format)
    agg["ess_per_sec"] = agg["ess_per_sec"].map("{:.1f}".format)

    print("Summary (mean over reps):")
    print(agg.to_string(index=False))
    print()

    make_plots(df)


if __name__ == "__main__":
    main()