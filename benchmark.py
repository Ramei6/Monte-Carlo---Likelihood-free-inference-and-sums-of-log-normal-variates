"""
benchmark.py
============
Compare Reject-ABC (Question 1) vs MCMC-ABC (Question 2) for the
sum-of-log-normals model across a shared ε grid.

Model
-----
    Y_i = Σ_{l=1}^L exp(X_{i,l}),   X_{i,l} ~ N(μ, σ²),   L=10,  n=1 000
    True parameters:  μ₀ = 0.0,  σ₀ = 0.3  (σ₀² = 0.09)
    Prior:            μ  ~ N(0, s²),   log(σ²) ~ N(0, t²),   s = t = 1.0
    Distance:         1-Wasserstein  W₁

Benchmark protocol
------------------
* R independent repetitions; each rep draws a fresh y_obs (n=1 000).
* Both algorithms see the SAME y_obs within every rep.
* SAME ε grid for both algorithms.
* Point estimate: posterior mean (of μ and σ²).
* Reject-ABC: keeps N_KEEP_REJ i.i.d. draws from the ABC posterior.
* MCMC-ABC  : runs one chain of N_BURN + N_ITER steps, drops burn-in.
              Stores (μ, σ) — σ is squared after the run → σ².

Key design choices
------------------
* Both functions are implemented directly in this file (no external imports
  from the project notebooks/modules) so the benchmark is self-contained.
* _mcmc_chain passes y_obs_sorted as a DYNAMIC argument (not baked into a
  closure), so JAX compiles the chain ONCE and reuses it across all reps
  and ε values.  Only n_total is static (it fixes array shapes).
* _batch_reject is similarly compiled once per (batch_size, n, L) triple.
* First-call timing includes JAX JIT compilation — subsequent calls are fast.

Outputs
-------
* benchmark_results/benchmark_results.csv  — raw records (rep × ε × method)
* benchmark_results/benchmark_plots.png    — 4-panel comparison figure
"""

import os
import time

import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
import matplotlib
matplotlib.use("Agg")          # headless-safe; remove if interactive display is wanted
import matplotlib.pyplot as plt
import pandas as pd

os.makedirs("benchmark_results", exist_ok=True)

# ─── Global constants ──────────────────────────────────────────────────────────
L           = 10
N_OBS       = 1_000
TRUE_MU     = 0.0
TRUE_SIGMA  = 0.3
TRUE_SIGMA2 = TRUE_SIGMA ** 2          # 0.09

S_PRIOR     = 1.0
T_PRIOR     = 1.0

# ── Benchmark settings ─────────────────────────────────────────────────────────
R           = 3                         # independent repetitions
EPS_GRID    = np.array([0.5, 0.7, 1.0, 1.5, 2.0, 3.0])

# ── Reject-ABC settings ────────────────────────────────────────────────────────
N_KEEP_REJ  = 200                       # accepted draws to collect
MAX_PROP    = 500_000                   # hard cap on proposals
BATCH_SIZE  = 4_096                     # vmap batch size for _batch_reject

# ── MCMC-ABC settings ──────────────────────────────────────────────────────────
DELTA       = 0.3                       # random-walk step (on μ and log σ)
N_BURN      = 2_000                     # burn-in steps (discarded)
N_ITER      = 3_000                     # post-burn-in steps (kept)
N_TOTAL     = N_BURN + N_ITER           # = 5 000


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  SHARED UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def generate_data(key, mu: float, sigma: float, n: int, l: int) -> jnp.ndarray:
    """Simulate n sums-of-log-normals: Y_i = Σ_l exp(μ + σ·Z), Z~N(0,1)."""
    Z = jax.random.normal(key, shape=(n, l))
    return jnp.sum(jnp.exp(mu + sigma * Z), axis=1)   # shape (n,)


@jax.jit
def w1(y_obs_sorted: jnp.ndarray, y_sim: jnp.ndarray) -> jnp.ndarray:
    """1-Wasserstein distance: mean |sorted gap| (both arrays length n)."""
    return jnp.mean(jnp.abs(y_obs_sorted - jnp.sort(y_sim)))


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  REJECT-ABC
# ═══════════════════════════════════════════════════════════════════════════════

@partial(jax.jit, static_argnums=(1, 4, 5))
def _batch_reject(
    key: jnp.ndarray,
    bs: int,
    s: float,
    t: float,
    n: int,
    l: int,
    y_obs_sorted: jnp.ndarray,
) -> tuple:
    """
    Draw bs proposals from the prior, simulate one dataset each, compute W₁.

    Static args: bs, n, l  (determine array shapes at compile time).
    Dynamic args: s, t, y_obs_sorted  (reuse cached XLA for different values).

    Returns
    -------
    mus     : (bs,)  — proposed μ values
    sigma2s : (bs,)  — proposed σ² values (= exp(t·η), η~N(0,1))
    dists   : (bs,)  — W₁ distances to y_obs_sorted
    """
    k_mu, k_eta, k_sim = jax.random.split(key, 3)
    mus     = s * jax.random.normal(k_mu,  shape=(bs,))
    sigma2s = jnp.exp(t * jax.random.normal(k_eta, shape=(bs,)))  # σ²=exp(t·η)

    def sim_one(k, mu, s2):
        Z = jax.random.normal(k, shape=(n, l))
        return jnp.sum(jnp.exp(mu + jnp.sqrt(s2) * Z), axis=1)

    Y_sims = jax.vmap(sim_one)(jax.random.split(k_sim, bs), mus, sigma2s)
    dists  = jax.vmap(lambda y: w1(y_obs_sorted, y))(Y_sims)
    return mus, sigma2s, dists


def run_reject_abc(
    key: jnp.ndarray,
    y_obs_sorted: jnp.ndarray,
    epsilon: float,
    n_keep: int = N_KEEP_REJ,
    s: float = S_PRIOR,
    t: float = T_PRIOR,
    max_proposals: int = MAX_PROP,
) -> tuple:
    """
    Reject-ABC.

    Parameters
    ----------
    key          : JAX PRNG key
    y_obs_sorted : pre-sorted observed data, shape (n,)
    epsilon      : ABC tolerance
    n_keep       : target number of accepted draws
    s, t         : prior hyperparameters
    max_proposals: hard cap (returns partial results with a warning)

    Returns
    -------
    accepted    : ndarray (n_accepted, 2) — columns [μ, σ²]
    acc_rate    : fraction of proposals accepted (= n_accepted / proposals)
    n_proposals : total proposals examined
    """
    n = len(y_obs_sorted)
    acc_mu: list = []
    acc_s2: list = []
    proposals = 0

    while len(acc_mu) < n_keep and proposals < max_proposals:
        bs        = min(BATCH_SIZE, max_proposals - proposals)
        key, sub  = jax.random.split(key)
        mus_b, s2_b, d_b = _batch_reject(
            sub, bs, float(s), float(t), n, L, y_obs_sorted
        )
        mu_np = np.array(mus_b)
        s2_np = np.array(s2_b)
        d_np  = np.array(d_b)

        idx       = np.flatnonzero(d_np <= epsilon)
        remaining = n_keep - len(acc_mu)

        if idx.size >= remaining:
            idx        = idx[:remaining]
            proposals += int(idx[-1]) + 1   # proposals examined up to last used
        else:
            proposals += bs

        acc_mu.extend(mu_np[idx].tolist())
        acc_s2.extend(s2_np[idx].tolist())

    if not acc_mu:
        return np.empty((0, 2)), 0.0, proposals

    accepted = np.column_stack([acc_mu, acc_s2])   # (n_accepted, 2)
    acc_rate = len(acc_mu) / proposals if proposals > 0 else 0.0

    if len(acc_mu) < n_keep:
        print(f"    [Reject-ABC ε={epsilon:.2f}] Warning: only "
              f"{len(acc_mu)}/{n_keep} draws accepted after {proposals} proposals.")

    return accepted, acc_rate, proposals


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  MCMC-ABC
# ═══════════════════════════════════════════════════════════════════════════════

def _log_prior(theta: jnp.ndarray) -> jnp.ndarray:
    """
    Log-prior for θ = (μ, σ).
    Prior: μ ~ N(0, s²),  log(σ²) = 2·log(σ) ~ N(0, t²).
    """
    mu, sigma = theta[0], theta[1]
    lp_mu    = -0.5 * (mu    / S_PRIOR) ** 2
    lp_lsig2 = -0.5 * (2.0 * jnp.log(sigma) / T_PRIOR) ** 2
    return lp_mu + lp_lsig2


def _propose(key: jnp.ndarray, theta: jnp.ndarray, delta: float) -> jnp.ndarray:
    """Symmetric random-walk proposal on (μ, log σ) — ratio cancels in M-H."""
    k1, k2    = jax.random.split(key)
    mu_new    = theta[0] + delta * jax.random.normal(k1)
    sigma_new = jnp.exp(jnp.log(theta[1]) + delta * jax.random.normal(k2))
    return jnp.array([mu_new, sigma_new])


def _simulate_n(key: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    """
    Simulate N_OBS sums-of-log-normals under θ=(μ,σ).
    N_OBS and L are global Python ints → concrete shapes at JIT trace time.
    """
    return jnp.sum(
        jnp.exp(theta[0] + theta[1] * jax.random.normal(key, shape=(N_OBS, L))),
        axis=1,
    )


def _mcmc_body(i: int, state: tuple) -> tuple:
    """
    One Metropolis-Hastings step of MCMC-ABC.

    Carry = (samples, θ_curr, key, n_acc, y_obs_sorted, ε, δ).
    Acceptance rule (Marjoram et al., 2003):
        accept  iff  W₁(y_sim, y_obs) ≤ ε  AND  log u < log[π(θ')/π(θ)]
    """
    samples, theta, key, n_acc, y_obs_s, eps, delta = state
    key, k_prop, k_sim, k_acc = jax.random.split(key, 4)

    theta_new = _propose(k_prop, theta, delta)
    y_sim     = _simulate_n(k_sim, theta_new)
    d         = w1(y_obs_s, y_sim)

    eps_ok    = (d <= eps)
    log_h     = _log_prior(theta_new) - _log_prior(theta)     # log prior ratio
    log_u     = jnp.log(jax.random.uniform(k_acc))
    accept    = eps_ok & (log_h > log_u)

    theta_out = jnp.where(accept, theta_new, theta)           # (2,)
    n_acc_out = n_acc + jnp.where(accept, 1, 0)
    samples   = samples.at[i].set(theta_out)

    return samples, theta_out, key, n_acc_out, y_obs_s, eps, delta


@partial(jax.jit, static_argnums=(4,))
def _mcmc_chain(
    key: jnp.ndarray,
    theta0: jnp.ndarray,
    y_obs_sorted: jnp.ndarray,
    epsilon: float,
    n_total: int,
    delta: float,
) -> tuple:
    """
    Run one MCMC-ABC chain of length n_total.

    n_total is STATIC (fixes jnp.zeros shape and fori_loop bounds).
    y_obs_sorted, epsilon, delta are DYNAMIC — no recompilation across reps/ε.

    Returns
    -------
    samples  : (n_total, 2) — full chain in (μ, σ) space
    acc_rate : scalar — fraction of proposed moves accepted
    """
    samples = jnp.zeros((n_total, 2))
    samples = samples.at[0].set(theta0)
    init    = (
        samples, theta0, key,
        jnp.array(0, dtype=jnp.int32),
        y_obs_sorted,
        jnp.array(epsilon, dtype=jnp.float32),
        jnp.array(delta,   dtype=jnp.float32),
    )
    samples, _, _, n_acc, _, _, _ = jax.lax.fori_loop(
        1, n_total, _mcmc_body, init
    )
    acc_rate = n_acc / jnp.array(n_total - 1, dtype=jnp.float32)
    return samples, acc_rate


def _find_valid_init(
    key: jnp.ndarray,
    y_obs_sorted: jnp.ndarray,
    epsilon: float,
    n_tries: int = 10_000,
) -> tuple:
    """
    Sample θ from the prior until W₁(y_sim(θ), y_obs) ≤ ε.
    Uses a Python loop (called once, before JIT chain).

    Returns
    -------
    theta0 : jnp.ndarray (2,) — valid starting point (μ, σ)
    key    : updated PRNG key
    """
    for _ in range(n_tries):
        key, k1, k2, k3 = jax.random.split(key, 4)
        mu_try    = float(S_PRIOR * jax.random.normal(k1))
        eta_try   = float(T_PRIOR * jax.random.normal(k2))   # log(σ²) ~ N(0, t²)
        sigma_try = float(jnp.exp(0.5 * eta_try))            # σ = exp(½·log σ²)
        theta_try = jnp.array([mu_try, sigma_try])
        y_try     = _simulate_n(k3, theta_try)
        if float(w1(y_obs_sorted, y_try)) <= epsilon:
            return theta_try, key
    raise RuntimeError(
        f"No valid θ₀ found in {n_tries} tries for ε={epsilon:.3f}. "
        f"Try increasing ε or n_tries."
    )


def run_mcmc_abc(
    key: jnp.ndarray,
    y_obs_sorted: jnp.ndarray,
    epsilon: float,
    n_burn: int = N_BURN,
    n_iter: int = N_ITER,
    delta:  float = DELTA,
) -> tuple:
    """
    MCMC-ABC (Marjoram et al., 2003).

    Returns
    -------
    post     : ndarray (n_iter, 2) — post-burn-in samples, columns [μ, σ²]
    acc_rate : float — full-chain MH acceptance rate (incl. burn-in)
    """
    theta0, key = _find_valid_init(key, y_obs_sorted, epsilon)
    n_total     = n_burn + n_iter

    samples, acc_rate = _mcmc_chain(
        key, theta0, y_obs_sorted,
        float(epsilon), n_total, float(delta),
    )

    post = np.array(samples)[n_burn:]    # (n_iter, 2),  cols = [μ, σ]
    post[:, 1] = post[:, 1] ** 2        # σ  →  σ²  (MCMC stores σ, not σ²)
    return post, float(acc_rate)


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  BENCHMARK LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def run_benchmark() -> pd.DataFrame:
    records   = []
    key_bench = jax.random.PRNGKey(2025)

    print("=" * 65)
    print("  Benchmark: Reject-ABC  vs  MCMC-ABC")
    print("=" * 65)
    print(f"  Model  : Y_i = Σ exp(X_il),  L={L},  n={N_OBS}")
    print(f"  Truth  : μ₀={TRUE_MU},  σ₀²={TRUE_SIGMA2}")
    print(f"  Prior  : μ~N(0,{S_PRIOR}²),  log(σ²)~N(0,{T_PRIOR}²)")
    print(f"  Reps   : R={R}")
    print(f"  ε grid : {EPS_GRID}")
    print(f"  Reject-ABC : n_keep={N_KEEP_REJ}, batch={BATCH_SIZE}, "
          f"max_prop={MAX_PROP}")
    print(f"  MCMC-ABC   : n_burn={N_BURN}, n_iter={N_ITER}, δ={DELTA}")
    print(f"  (First JAX call compiles; subsequent calls reuse cached XLA)\n")

    for r in range(R):
        key_bench, k_data = jax.random.split(key_bench)
        y_obs        = generate_data(k_data, TRUE_MU, TRUE_SIGMA, N_OBS, L)
        y_obs_sorted = jnp.sort(y_obs)

        print(f"─── Rep {r+1}/{R}  "
              f"(mean={float(jnp.mean(y_obs)):.2f}, "
              f"std={float(jnp.std(y_obs)):.2f}) ───")

        for eps in EPS_GRID:
            key_bench, k_rej, k_mcmc = jax.random.split(key_bench, 3)

            # ── Reject-ABC ────────────────────────────────────────────────────
            t0 = time.perf_counter()
            try:
                acc, rate_rej, n_prop = run_reject_abc(
                    k_rej, y_obs_sorted, float(eps)
                )
                rt_rej = time.perf_counter() - t0
                if len(acc) == 0:
                    mean_mu_rej, mean_s2_rej, rate_rej, n_acc_rej = (
                        float("nan"), float("nan"), 0.0, 0
                    )
                else:
                    mean_mu_rej = float(np.mean(acc[:, 0]))
                    mean_s2_rej = float(np.mean(acc[:, 1]))
                    n_acc_rej   = len(acc)
            except Exception as exc:
                print(f"    [Reject-ABC ε={eps:.2f}] ERROR: {exc}")
                rt_rej = float("nan")
                mean_mu_rej, mean_s2_rej, rate_rej, n_acc_rej = (
                    float("nan"), float("nan"), float("nan"), 0
                )

            records.append(dict(
                rep=r, epsilon=float(eps), method="Reject-ABC",
                runtime_s=rt_rej, mean_mu=mean_mu_rej,
                mean_sigma2=mean_s2_rej, acc_rate=rate_rej,
                n_accepted=n_acc_rej,
            ))

            # ── MCMC-ABC ──────────────────────────────────────────────────────
            t0 = time.perf_counter()
            try:
                post, rate_mcmc = run_mcmc_abc(k_mcmc, y_obs_sorted, float(eps))
                rt_mcmc = time.perf_counter() - t0
                mean_mu_mcmc  = float(np.mean(post[:, 0]))
                mean_s2_mcmc  = float(np.mean(post[:, 1]))
                n_acc_mcmc    = int(rate_mcmc * (N_BURN + N_ITER - 1))
            except Exception as exc:
                print(f"    [MCMC-ABC ε={eps:.2f}] ERROR: {exc}")
                rt_mcmc = float("nan")
                mean_mu_mcmc, mean_s2_mcmc, rate_mcmc, n_acc_mcmc = (
                    float("nan"), float("nan"), float("nan"), 0
                )

            records.append(dict(
                rep=r, epsilon=float(eps), method="MCMC-ABC",
                runtime_s=rt_mcmc, mean_mu=mean_mu_mcmc,
                mean_sigma2=mean_s2_mcmc, acc_rate=rate_mcmc,
                n_accepted=n_acc_mcmc,
            ))

            print(
                f"  ε={eps:.1f} | "
                f"Reject: t={rt_rej:6.1f}s  acc={rate_rej:.4f}  "
                f"E[μ]={mean_mu_rej:.3f}  E[σ²]={mean_s2_rej:.4f} | "
                f"MCMC: t={rt_mcmc:6.1f}s  acc={rate_mcmc:.4f}  "
                f"E[μ]={mean_mu_mcmc:.3f}  E[σ²]={mean_s2_mcmc:.4f}"
            )

        print()

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

def make_plots(df: pd.DataFrame) -> None:
    """6-panel comparison with error bands, log-scale runtime, and speedup ratio."""

    grp  = df.groupby(["method", "epsilon"])
    mean = grp.mean(numeric_only=True).reset_index()
    std  = grp.std(numeric_only=True).reset_index()

    def _get(method):
        m = mean[mean.method == method].sort_values("epsilon").reset_index(drop=True)
        s = std[std.method   == method].sort_values("epsilon").reset_index(drop=True)
        return m, s

    R_m, R_s = _get("Reject-ABC")
    M_m, M_s = _get("MCMC-ABC")

    C_R, C_M = "#C0392B", "#2471A3"     # red = Reject-ABC, blue = MCMC-ABC
    eps = R_m.epsilon.values

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        r"Reject-ABC  vs  MCMC-ABC — Comparison across $\varepsilon$"
        "\n"
        r"model: $Y_i=\!\sum_\ell e^{X_{i,\ell}}$, "
        r"$\mu_0=0$, $\sigma_0^2=0.09$, $n=1\,000$, $L=10$"
        f"\n(mean \u00b1 1 std over {R} repetitions)",
        fontsize=12,
    )

    def _band(ax, x, m_col, s_col, m_df, s_df, color, marker, ls, label):
        y  = m_df[m_col].values
        dy = s_df[s_col].values
        ax.plot(x, y, marker=marker, ls=ls, color=color, lw=2, ms=6, label=label)
        ax.fill_between(x, y - dy, y + dy, color=color, alpha=0.15)

    # ── (a) Runtime — log scale ───────────────────────────────────────────────
    ax = axes[0, 0]
    _band(ax, eps, "runtime_s", "runtime_s", R_m, R_s, C_R, "o", "-",  "Reject-ABC")
    _band(ax, eps, "runtime_s", "runtime_s", M_m, M_s, C_M, "s", "--", "MCMC-ABC")
    ax.set_yscale("log")
    ax.set_xlabel(r"Tolerance $\varepsilon$")
    ax.set_ylabel("Wall-clock time (s)  [log scale]")
    ax.set_title(r"(a) Runtime vs $\varepsilon$")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    # ── (b) Speedup ratio Reject / MCMC ──────────────────────────────────────
    ax = axes[0, 1]
    ratio = R_m["runtime_s"].values / M_m["runtime_s"].values
    bars  = ax.bar(eps, ratio, width=0.18, color=C_R, alpha=0.75, edgecolor="white")
    ax.axhline(1, color="#555", lw=1.2, ls="--", label="Ratio = 1  (equal speed)")
    for bar, v in zip(bars, ratio):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.3,
                f"{v:.1f}×", ha="center", va="bottom", fontsize=8.5, color=C_R)
    ax.set_xlabel(r"Tolerance $\varepsilon$")
    ax.set_ylabel(r"$t_{\rm Reject} \;/\; t_{\rm MCMC}$")
    ax.set_title("(b) Speedup ratio (Reject / MCMC)")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    # ── (c) Posterior mean μ ──────────────────────────────────────────────────
    ax = axes[0, 2]
    _band(ax, eps, "mean_mu", "mean_mu", R_m, R_s, C_R, "o", "-",  "Reject-ABC")
    _band(ax, eps, "mean_mu", "mean_mu", M_m, M_s, C_M, "s", "--", "MCMC-ABC")
    ax.axhline(TRUE_MU, color="#555", lw=1.4, ls=":", label=r"$\mu_0 = 0.0$")
    ax.set_xlabel(r"Tolerance $\varepsilon$")
    ax.set_ylabel(r"$\hat{\mu} = \mathbb{E}[\mu \mid y^\star]$")
    ax.set_title(r"(c) Posterior mean $\mu$ vs $\varepsilon$")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── (d) Posterior mean σ² ─────────────────────────────────────────────────
    ax = axes[1, 0]
    _band(ax, eps, "mean_sigma2", "mean_sigma2", R_m, R_s, C_R, "o", "-",  "Reject-ABC")
    _band(ax, eps, "mean_sigma2", "mean_sigma2", M_m, M_s, C_M, "s", "--", "MCMC-ABC")
    ax.axhline(TRUE_SIGMA2, color="#555", lw=1.4, ls=":", label=r"$\sigma_0^2 = 0.09$")
    ax.set_xlabel(r"Tolerance $\varepsilon$")
    ax.set_ylabel(r"$\widehat{\sigma^2} = \mathbb{E}[\sigma^2 \mid y^\star]$")
    ax.set_title(r"(d) Posterior mean $\sigma^2$ vs $\varepsilon$")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── (e) Acceptance rate ───────────────────────────────────────────────────
    ax = axes[1, 1]
    _band(ax, eps, "acc_rate", "acc_rate", R_m, R_s, C_R, "o", "-",  "Reject-ABC")
    _band(ax, eps, "acc_rate", "acc_rate", M_m, M_s, C_M, "s", "--", "MCMC-ABC")
    ax.set_xlabel(r"Tolerance $\varepsilon$")
    ax.set_ylabel("Acceptance rate")
    ax.set_title(r"(e) Acceptance rate vs $\varepsilon$")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── (f) Absolute bias for μ and σ² ───────────────────────────────────────
    ax = axes[1, 2]
    bias_mu_R  = (R_m["mean_mu"]     - TRUE_MU).abs().values
    bias_mu_M  = (M_m["mean_mu"]     - TRUE_MU).abs().values
    bias_s2_R  = (R_m["mean_sigma2"] - TRUE_SIGMA2).abs().values
    bias_s2_M  = (M_m["mean_sigma2"] - TRUE_SIGMA2).abs().values
    ax.plot(eps, bias_mu_R,  "o-",   color=C_R, lw=2, ms=6,  label=r"Reject $|\hat\mu - \mu_0|$")
    ax.plot(eps, bias_mu_M,  "s--",  color=C_M, lw=2, ms=6,  label=r"MCMC $|\hat\mu - \mu_0|$")
    ax.plot(eps, bias_s2_R,  "o:",   color=C_R, lw=2, ms=6,
            label=r"Reject $|\widehat{\sigma^2} - \sigma_0^2|$")
    ax.plot(eps, bias_s2_M,  "s:",   color=C_M, lw=2, ms=6,
            label=r"MCMC $|\widehat{\sigma^2} - \sigma_0^2|$")
    ax.set_xlabel(r"Tolerance $\varepsilon$")
    ax.set_ylabel("Absolute bias")
    ax.set_title(r"(f) Bias vs $\varepsilon$  (solid $= \mu$, dotted $= \sigma^2$)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = "benchmark_results/benchmark_plots.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plots saved  → {plot_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    df = run_benchmark()

    # ── Save raw results ─────────────────────────────────────────────────────
    csv_path = "benchmark_results/benchmark_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Results saved → {csv_path}\n")

    # ── Pretty-print summary table ───────────────────────────────────────────
    agg = (
        df.groupby(["method", "epsilon"])
        .mean(numeric_only=True)[["runtime_s", "mean_mu", "mean_sigma2", "acc_rate"]]
        .reset_index()
    )
    agg.columns = ["method", "epsilon", "runtime_s", "mean_mu", "mean_sigma2", "acc_rate"]
    agg["mean_mu"]    = agg["mean_mu"].map("{:.4f}".format)
    agg["mean_sigma2"] = agg["mean_sigma2"].map("{:.5f}".format)
    agg["acc_rate"]   = agg["acc_rate"].map("{:.4f}".format)
    agg["runtime_s"]  = agg["runtime_s"].map("{:.2f}".format)
    print("Summary (averaged over reps):")
    print(agg.to_string(index=False))
    print()

    # ── Plots ────────────────────────────────────────────────────────────────
    make_plots(df)


if __name__ == "__main__":
    main()
