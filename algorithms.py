"""
algorithms.py
=============
Minimal standalone implementations of Reject-ABC and MCMC-ABC for the
sum-of-log-normals model.  K = 1 throughout (single simulation per
proposal / step) — the simplest comparable versions.

Internal MCMC parametrisation: θ = (μ, log σ²) — unconstrained.
All public functions return samples as (μ_array, σ²_array) numpy arrays.

Public API
----------
    generate_dataset(key, mu, sigma2, n, l)  →  jax array (n,)
    reject_abc(y_obs, epsilon, ...)          →  mu, sigma2, acc_rate, ess
    mcmc_abc(y_obs, epsilon, ...)            →  mu, sigma2, acc_rate, ess
    compute_ess(x)                           →  float
    warmup(n_obs, eps)                       →  (pre-compiles JAX kernels)
"""

import functools
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

# ── Model constants ───────────────────────────────────────────────────────────
L           = 10        # log-normals per observation
TRUE_MU     = 0.0
TRUE_SIGMA2 = 0.09      # = (0.3)²

# ── Algorithm defaults ────────────────────────────────────────────────────────
S_PRIOR     = 1.0       # μ  ~ N(0, S²)
T_PRIOR     = 1.0       # log σ² ~ N(0, T²)

N_KEEP_REJ  = 300       # Reject-ABC: target accepted draws
BATCH_SIZE  = 4_096     # Reject-ABC: vmap batch size
MAX_PROP    = 1_000_000 # Reject-ABC: hard proposal cap

DELTA_MCMC  = 0.2       # MCMC-ABC: random-walk step (near-optimal from Q2)
N_BURN_MCMC = 5_000     # MCMC-ABC: burn-in iterations  (discarded)
N_ITER_MCMC = 15_000    # MCMC-ABC: post-burn-in iterations  (kept)


# ═════════════════════════════════════════════════════════════════════════════
# Shared model primitive
# ═════════════════════════════════════════════════════════════════════════════

def generate_dataset(key, mu=TRUE_MU, sigma2=TRUE_SIGMA2, n=1_000, l=L):
    """Simulate n observations  Y_i = Σ_l exp(X_{i,l}),  X ~ N(μ, σ²)."""
    X = mu + jnp.sqrt(sigma2) * jax.random.normal(key, shape=(n, l))
    return jnp.sum(jnp.exp(X), axis=1)


# ═════════════════════════════════════════════════════════════════════════════
# ESS utility
# ═════════════════════════════════════════════════════════════════════════════

def compute_ess(x):
    """
    Effective sample size via truncated ACF.

    ESS = n / (1 + 2 Σ_k ρ_k),  summing until the first negative lag.
    For i.i.d. samples every ρ_k = 0, so ESS = n exactly.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 4:
        return float(n)
    xm  = x - x.mean()
    var = np.dot(xm, xm) / n
    if var == 0.0:
        return 1.0
    acf_sum = 0.0
    for k in range(1, min(500, n // 4) + 1):
        rho = np.dot(xm[k:], xm[:n - k]) / (n * var)
        if rho < 0.0:
            break
        acf_sum += rho
    return float(np.clip(n / (1.0 + 2.0 * acf_sum), 1.0, float(n)))


# ═════════════════════════════════════════════════════════════════════════════
# Algorithm 1 — Reject-ABC
# ═════════════════════════════════════════════════════════════════════════════

@partial(jax.jit, static_argnums=(1, 4, 5))
def _batch_rej(key, bs, s, t, n, l, y_sorted):
    """
    Draw bs proposals from prior, simulate one dataset each (K=1), return W₁.

    Static args: bs, n, l — fix array shapes at compile time so that JAX
    reuses this compiled kernel for all epsilon values.
    """
    k_mu, k_eta, k_sim = jax.random.split(key, 3)
    mus     = s * jax.random.normal(k_mu,  shape=(bs,))
    sigma2s = jnp.exp(t * jax.random.normal(k_eta, shape=(bs,)))

    def sim_one(k, mu, s2):
        Z = jax.random.normal(k, shape=(n, l))
        y = jnp.sum(jnp.exp(mu + jnp.sqrt(s2) * Z), axis=1)
        return jnp.mean(jnp.abs(y_sorted - jnp.sort(y)))

    dists = jax.vmap(sim_one)(jax.random.split(k_sim, bs), mus, sigma2s)
    return mus, sigma2s, dists


def reject_abc(y_obs, epsilon,
               n_keep=N_KEEP_REJ, s=S_PRIOR, t=T_PRIOR,
               seed=0, max_proposals=MAX_PROP):
    """
    Reject-ABC with K=1 (one simulation per proposal).

    Parameters
    ----------
    y_obs    : array-like (n,) — observed data
    epsilon  : ABC tolerance
    n_keep   : target number of accepted draws
    s, t     : prior hyperparameters
    seed     : JAX PRNG seed
    max_proposals : hard proposal cap

    Returns
    -------
    mu_post     : (n_keep,) numpy array of accepted μ
    sigma2_post : (n_keep,) numpy array of accepted σ²
    acc_rate    : float  — accepted / total proposals
    ess         : float  — n_keep  (draws are i.i.d. by construction)
    """
    key = jax.random.PRNGKey(seed)
    n   = len(y_obs)
    ys  = jnp.sort(jnp.asarray(y_obs))

    mu_acc, s2_acc = [], []
    n_prop = 0

    while len(mu_acc) < n_keep and n_prop < max_proposals:
        bs       = min(BATCH_SIZE, max_proposals - n_prop)
        key, sub = jax.random.split(key)
        mu_b, s2_b, d_b = _batch_rej(sub, bs, float(s), float(t), n, L, ys)

        d_np = np.asarray(d_b)
        idx  = np.flatnonzero(d_np <= epsilon)
        need = n_keep - len(mu_acc)

        if idx.size >= need:
            idx     = idx[:need]
            n_prop += int(idx[-1]) + 1
        else:
            n_prop += bs

        mu_acc.extend(np.asarray(mu_b)[idx].tolist())
        s2_acc.extend(np.asarray(s2_b)[idx].tolist())

    mu_post     = np.array(mu_acc[:n_keep])
    sigma2_post = np.array(s2_acc[:n_keep])
    n_got       = len(mu_post)
    acc_rate    = n_got / n_prop if n_prop > 0 else 0.0

    if n_got < n_keep:
        print(f"  [Reject-ABC ε={epsilon:.3f}] Warning: only "
              f"{n_got}/{n_keep} accepted after {n_prop} proposals.")

    return mu_post, sigma2_post, acc_rate, float(n_got)  # ESS = n_keep (i.i.d.)


# ═════════════════════════════════════════════════════════════════════════════
# Algorithm 2 — MCMC-ABC
# ═════════════════════════════════════════════════════════════════════════════

@functools.lru_cache(maxsize=16)
def _make_chain_fn(n, n_total, s=S_PRIOR, t=T_PRIOR):
    """
    Factory: returns a JIT-compiled single-chain MCMC-ABC runner.

    Cached via lru_cache — the same compiled function is reused for all
    (ε, δ) values as long as (n, n_total, s, t) stay the same.  The cache
    avoids repeated XLA recompilation across the epsilon grid.

    The chain length n_total and dataset size n are captured as Python ints
    in the closure, which makes shape=(n, L) and length=n_total concrete at
    JAX trace time.
    """
    @jax.jit
    def run(y_sorted, theta0, key, eps, delta):
        """
        Run one MCMC-ABC chain of n_total steps.

        State: θ = (μ, log σ²)  —  unconstrained for symmetric RW proposals.
        Each step stacks the output as (μ, σ²) for easy post-processing.
        """
        def step(carry, _):
            theta, rng, n_acc = carry
            rng, kp1, kp2, ks, ku = jax.random.split(rng, 5)

            mu, lsig2 = theta[0], theta[1]

            # Symmetric random-walk proposal on (μ, log σ²)
            mu_new    = mu    + delta * jax.random.normal(kp1)
            lsig2_new = lsig2 + delta * jax.random.normal(kp2)

            # Simulate K=1 dataset under θ_new
            sigma_new = jnp.exp(0.5 * lsig2_new)
            Z   = jax.random.normal(ks, shape=(n, L))   # n, L: Python ints
            y_s = jnp.sum(jnp.exp(mu_new + sigma_new * Z), axis=1)
            d   = jnp.mean(jnp.abs(y_sorted - jnp.sort(y_s)))

            # MH acceptance: ABC gate AND prior ratio
            lp_new = -0.5 * (mu_new / s) ** 2 - 0.5 * (lsig2_new / t) ** 2
            lp_old = -0.5 * (mu    / s) ** 2 - 0.5 * (lsig2     / t) ** 2
            accept = (d <= eps) & (
                (lp_new - lp_old) >= jnp.log(jax.random.uniform(ku))
            )

            theta_next = jnp.where(accept, jnp.stack([mu_new, lsig2_new]), theta)
            out = jnp.stack([theta_next[0], jnp.exp(theta_next[1])])  # (μ, σ²)
            return (theta_next, rng, n_acc + accept.astype(jnp.int32)), out

        (_, _, n_acc), samples = jax.lax.scan(
            step,
            (theta0, key, jnp.array(0, dtype=jnp.int32)),
            xs=None,
            length=n_total,     # Python int from closure → static at trace time
        )
        return samples, n_acc / jnp.float32(n_total)

    return run


def _find_valid_init(y_sorted_np, epsilon, n, s=S_PRIOR, t=T_PRIOR,
                     seed=0, n_tries=20_000):
    """
    Sample θ ~ prior until W₁(y_sim(θ), y_obs) ≤ ε.

    Uses a plain numpy loop (called once per run, never JIT-compiled).
    Returns θ₀ = (μ, log σ²) as a float32 JAX array.
    """
    rng = np.random.default_rng(seed)
    for _ in range(n_tries):
        mu    = rng.normal() * s
        lsig2 = rng.normal() * t
        sigma = np.exp(0.5 * lsig2)
        Z     = rng.standard_normal((n, L))
        y_sim = np.sum(np.exp(mu + sigma * Z), axis=1)
        d     = np.mean(np.abs(np.sort(y_sim) - y_sorted_np))
        if d <= epsilon:
            return jnp.array([mu, lsig2], dtype=jnp.float32)
    raise RuntimeError(
        f"No valid θ₀ found in {n_tries} tries for ε={epsilon:.3f}. "
        "Try a larger ε or increase n_tries."
    )


def mcmc_abc(y_obs, epsilon,
             delta=DELTA_MCMC, n_iter=N_ITER_MCMC, n_burn=N_BURN_MCMC,
             s=S_PRIOR, t=T_PRIOR, seed=0):
    """
    MCMC-ABC (Marjoram et al., 2003) with K=1, single chain.

    Parameters
    ----------
    y_obs            : array-like (n,)
    epsilon          : ABC tolerance
    delta            : random-walk step size on (μ, log σ²)
    n_iter / n_burn  : post-burn-in / burn-in iterations
    s, t             : prior hyperparameters
    seed             : integer seed for numpy init search and JAX chain

    Returns
    -------
    mu_post     : (n_iter,) numpy array of post-burn-in μ
    sigma2_post : (n_iter,) numpy array of post-burn-in σ²
    acc_rate    : float  — MH acceptance rate over the full chain
    ess         : float  — ESS of μ via truncated ACF
    """
    n       = len(y_obs)
    ys      = jnp.sort(jnp.asarray(y_obs))
    n_total = n_burn + n_iter

    theta0   = _find_valid_init(np.asarray(ys), epsilon, n, s=s, t=t, seed=seed)
    chain_fn = _make_chain_fn(n, n_total, s, t)

    key = jax.random.PRNGKey(seed + 7777)
    samples, acc_rate = chain_fn(
        ys,
        theta0,
        key,
        jnp.float32(epsilon),
        jnp.float32(delta),
    )

    post        = np.asarray(samples)[n_burn:]  # (n_iter, 2): cols = (μ, σ²)
    mu_post     = post[:, 0]
    sigma2_post = post[:, 1]
    ess         = compute_ess(mu_post)
    return mu_post, sigma2_post, float(acc_rate), ess


# ═════════════════════════════════════════════════════════════════════════════
# Warm-up utility
# ═════════════════════════════════════════════════════════════════════════════

def warmup(n_obs=1_000, eps=1.0):
    """
    Pre-compile JAX kernels.  Call once before the timed benchmark loop.

    - Compiles _batch_rej for (BATCH_SIZE, n_obs, L).
    - Compiles and runs the MCMC chain for (n_obs, N_BURN_MCMC + N_ITER_MCMC).
      This one full chain run is unavoidable: JAX JIT compilation happens
      on the first actual call, not before it.
    """
    key  = jax.random.PRNGKey(42)
    y_wup = jnp.ones(n_obs, dtype=jnp.float32)
    ys    = jnp.sort(y_wup)

    # Compile Reject-ABC batch kernel
    out = _batch_rej(key, BATCH_SIZE, S_PRIOR, T_PRIOR, n_obs, L, ys)
    jax.block_until_ready(out)

    # Compile + run MCMC chain (the first call triggers XLA compilation)
    theta0   = jnp.array([TRUE_MU, float(np.log(TRUE_SIGMA2))], dtype=jnp.float32)
    chain_fn = _make_chain_fn(n_obs, N_BURN_MCMC + N_ITER_MCMC)
    samples, _ = chain_fn(ys, theta0, key, jnp.float32(eps), jnp.float32(DELTA_MCMC))
    jax.block_until_ready(samples)
