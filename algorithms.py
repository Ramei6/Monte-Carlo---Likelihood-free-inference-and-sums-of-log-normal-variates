"""
algorithms.py

Implementations of Reject-ABC and MCMC-ABC for the sum-of-log-normals model.

Both methods use K = 1, meaning one simulated dataset per proposal or MCMC step.
For MCMC-ABC, the chain is run on theta = (mu, log sigma2), so the variance
parameter stays positive after transforming back.

Main functions:
    generate_dataset(key, mu, sigma2, n, l)  ->  simulated data
    reject_abc(y_obs, epsilon, ...)          ->  mu, sigma2, acc_rate, ess
    mcmc_abc(y_obs, epsilon, ...)            ->  mu, sigma2, acc_rate, ess
    compute_ess(x)                           ->  effective sample size
    warmup(n_obs, eps)                       ->  warm up JAX compilation
"""

"""Reject-ABC and MCMC-ABC code for the sum-of-log-normals model."""

import functools
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp


# Model constants
L = 10
TRUE_MU = 0.0
TRUE_SIGMA2 = 0.09

# Algorithm defaults
S_PRIOR = 1.0
T_PRIOR = 1.0

N_KEEP_REJ = 300
BATCH_SIZE = 4_096
MAX_PROP = 1_000_000

DELTA_MCMC = 0.2
N_BURN_MCMC = 5_000
N_ITER_MCMC = 15_000


def generate_dataset(key, mu=TRUE_MU, sigma2=TRUE_SIGMA2, n=1_000, l=L):
    """Simulate data from the sum-of-log-normals model."""
    x = mu + jnp.sqrt(sigma2) * jax.random.normal(key, shape=(n, l))
    return jnp.sum(jnp.exp(x), axis=1)


def compute_ess(x):
    """Estimate ESS using the autocorrelation sequence."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)

    if n < 4:
        return float(n)

    xm = x - x.mean()
    var = np.dot(xm, xm) / n

    if var == 0.0:
        return 1.0

    acf_sum = 0.0

    for k in range(1, min(500, n // 4) + 1):
        rho = np.dot(xm[k:], xm[:n - k]) / (n * var)

        if rho < 0.0:
            break

        acf_sum += rho

    ess = n / (1.0 + 2.0 * acf_sum)
    return float(np.clip(ess, 1.0, float(n)))


# Reject-ABC

@partial(jax.jit, static_argnums=(1, 4, 5))
def _batch_rej(key, bs, s, t, n, l, y_sorted):
    """Draw prior proposals and compute ABC distances."""
    k_mu, k_eta, k_sim = jax.random.split(key, 3)

    mus = s * jax.random.normal(k_mu, shape=(bs,))
    sigma2s = jnp.exp(t * jax.random.normal(k_eta, shape=(bs,)))

    def sim_one(k, mu, s2):
        z = jax.random.normal(k, shape=(n, l))
        y = jnp.sum(jnp.exp(mu + jnp.sqrt(s2) * z), axis=1)
        return jnp.mean(jnp.abs(y_sorted - jnp.sort(y)))

    dists = jax.vmap(sim_one)(jax.random.split(k_sim, bs), mus, sigma2s)
    return mus, sigma2s, dists


def reject_abc(
    y_obs,
    epsilon,
    n_keep=N_KEEP_REJ,
    s=S_PRIOR,
    t=T_PRIOR,
    seed=0,
    max_proposals=MAX_PROP,
):
    """Run rejection ABC and return accepted draws, acceptance rate, and ESS."""
    key = jax.random.PRNGKey(seed)
    n = len(y_obs)
    ys = jnp.sort(jnp.asarray(y_obs))

    mu_acc = []
    s2_acc = []
    n_prop = 0

    while len(mu_acc) < n_keep and n_prop < max_proposals:
        bs = min(BATCH_SIZE, max_proposals - n_prop)
        key, sub = jax.random.split(key)

        mu_b, s2_b, d_b = _batch_rej(
            sub,
            bs,
            float(s),
            float(t),
            n,
            L,
            ys,
        )

        d_np = np.asarray(d_b)
        idx = np.flatnonzero(d_np <= epsilon)
        need = n_keep - len(mu_acc)

        if idx.size >= need:
            idx = idx[:need]
            n_prop += int(idx[-1]) + 1
        else:
            n_prop += bs

        mu_acc.extend(np.asarray(mu_b)[idx].tolist())
        s2_acc.extend(np.asarray(s2_b)[idx].tolist())

    mu_post = np.array(mu_acc[:n_keep])
    sigma2_post = np.array(s2_acc[:n_keep])

    n_got = len(mu_post)
    acc_rate = n_got / n_prop if n_prop > 0 else 0.0

    if n_got < n_keep:
        print(
            f"  [Reject-ABC eps={epsilon:.3f}] Warning: only "
            f"{n_got}/{n_keep} accepted after {n_prop} proposals."
        )

    return mu_post, sigma2_post, acc_rate, float(n_got)


# MCMC-ABC

@functools.lru_cache(maxsize=16)
def _make_chain_fn(n, n_total, s=S_PRIOR, t=T_PRIOR):
    """Create the JIT-compiled MCMC chain function."""

    @jax.jit
    def run(y_sorted, theta0, key, eps, delta):
        """Run one MCMC-ABC chain."""

        def step(carry, _):
            theta, rng, n_acc = carry
            rng, kp1, kp2, ks, ku = jax.random.split(rng, 5)

            mu, lsig2 = theta[0], theta[1]

            # Random-walk proposal on (mu, log sigma2)
            mu_new = mu + delta * jax.random.normal(kp1)
            lsig2_new = lsig2 + delta * jax.random.normal(kp2)

            sigma_new = jnp.exp(0.5 * lsig2_new)
            z = jax.random.normal(ks, shape=(n, L))
            y_s = jnp.sum(jnp.exp(mu_new + sigma_new * z), axis=1)
            d = jnp.mean(jnp.abs(y_sorted - jnp.sort(y_s)))

            # ABC condition plus prior ratio
            lp_new = -0.5 * (mu_new / s) ** 2 - 0.5 * (lsig2_new / t) ** 2
            lp_old = -0.5 * (mu / s) ** 2 - 0.5 * (lsig2 / t) ** 2

            accept = (d <= eps) & (
                (lp_new - lp_old) >= jnp.log(jax.random.uniform(ku))
            )

            theta_next = jnp.where(
                accept,
                jnp.stack([mu_new, lsig2_new]),
                theta,
            )

            out = jnp.stack([theta_next[0], jnp.exp(theta_next[1])])

            return (
                theta_next,
                rng,
                n_acc + accept.astype(jnp.int32),
            ), out

        (_, _, n_acc), samples = jax.lax.scan(
            step,
            (theta0, key, jnp.array(0, dtype=jnp.int32)),
            xs=None,
            length=n_total,
        )

        return samples, n_acc / jnp.float32(n_total)

    return run


def _find_valid_init(
    y_sorted_np,
    epsilon,
    n,
    s=S_PRIOR,
    t=T_PRIOR,
    seed=0,
    n_tries=20_000,
):
    """Find an initial point accepted by the ABC tolerance."""
    rng = np.random.default_rng(seed)

    for _ in range(n_tries):
        mu = rng.normal() * s
        lsig2 = rng.normal() * t
        sigma = np.exp(0.5 * lsig2)

        z = rng.standard_normal((n, L))
        y_sim = np.sum(np.exp(mu + sigma * z), axis=1)
        d = np.mean(np.abs(np.sort(y_sim) - y_sorted_np))

        if d <= epsilon:
            return jnp.array([mu, lsig2], dtype=jnp.float32)

    raise RuntimeError(
        f"No valid initial point found in {n_tries} tries for eps={epsilon:.3f}. "
        "Try a larger epsilon or increase n_tries."
    )


def mcmc_abc(
    y_obs,
    epsilon,
    delta=DELTA_MCMC,
    n_iter=N_ITER_MCMC,
    n_burn=N_BURN_MCMC,
    s=S_PRIOR,
    t=T_PRIOR,
    seed=0,
):
    """Run one MCMC-ABC chain and return post-burn-in samples."""
    n = len(y_obs)
    ys = jnp.sort(jnp.asarray(y_obs))
    n_total = n_burn + n_iter

    theta0 = _find_valid_init(
        np.asarray(ys),
        epsilon,
        n,
        s=s,
        t=t,
        seed=seed,
    )

    chain_fn = _make_chain_fn(n, n_total, s, t)

    key = jax.random.PRNGKey(seed + 7777)

    samples, acc_rate = chain_fn(
        ys,
        theta0,
        key,
        jnp.float32(epsilon),
        jnp.float32(delta),
    )

    post = np.asarray(samples)[n_burn:]

    mu_post = post[:, 0]
    sigma2_post = post[:, 1]
    ess = compute_ess(mu_post)

    return mu_post, sigma2_post, float(acc_rate), ess


def warmup(n_obs=1_000, eps=1.0):
    """Run a warmup so JAX compilation is not included in benchmark timing."""
    key = jax.random.PRNGKey(42)

    y_wup = jnp.ones(n_obs, dtype=jnp.float32)
    ys = jnp.sort(y_wup)

    out = _batch_rej(
        key,
        BATCH_SIZE,
        S_PRIOR,
        T_PRIOR,
        n_obs,
        L,
        ys,
    )
    jax.block_until_ready(out)

    theta0 = jnp.array(
        [TRUE_MU, float(np.log(TRUE_SIGMA2))],
        dtype=jnp.float32,
    )

    chain_fn = _make_chain_fn(n_obs, N_BURN_MCMC + N_ITER_MCMC)

    samples, _ = chain_fn(
        ys,
        theta0,
        key,
        jnp.float32(eps),
        jnp.float32(DELTA_MCMC),
    )

    jax.block_until_ready(samples)
