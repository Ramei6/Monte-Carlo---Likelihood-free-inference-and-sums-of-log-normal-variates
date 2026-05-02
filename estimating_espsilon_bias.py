import jax
import jax.numpy as jnp
import jax.random as jr
from jax.nn import softplus
from jax.scipy.special import logsumexp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from functools import partial

## ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES GLOBALES
# Doivent être définies AVANT toute fonction (valeurs par défaut d'arguments).
# ══════════════════════════════════════════════════════════════════════════════
 
TRUE_MU    = 0.0
TRUE_SIGMA = 0.3
TRUE_SIG2  = TRUE_SIGMA ** 2   # 0.09  — espace Gibbs
 
L     = 10
M_OBS = 1_000
M_SIM = M_OBS
 
S_PRIOR = 1.0
T_PRIOR = 1.0
DELTA   = 0.3
 
N_BURN     = 20_000    # alias pour la signature interne de mcmc_abc_single
N_ITER     = 100_000   # idem
 
N_BURN_ABC  = N_BURN
N_ITER_ABC  = N_ITER
N_CHAINS    = 5
K_THIN      = 100      # 1 sample gardé sur K_THIN → 1 000 par chaîne
 
N_ITER_GIBBS = 10_000
N_BURN_GIBBS = 2_000
K_GIBBS      = 10
 
N_DATASETS   = 10
EPSILON_GRID = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 5.0]
 
SEED       = 42
OUTPUT_DIR = "epsilon_comparison"

###########################################################################################################################################
#                                                                                                                                         #
# ---------------------------------------------------- Dataset creation ------------------------------------------------------------------#
#                                                                                                                                         #
###########################################################################################################################################
def create_Y(key, n, L, theta):
    mu, sig2 = theta
    X = jnp.exp(mu + jnp.sqrt(sig2)*jax.random.normal(key, shape=(n,L)))
    Y = jnp.sum(X, axis=1)
    return Y

#key = jax.random.PRNGKey(0)
#theta = 0., 0.3
#L = 10
#n = 10000

#Y = create_Y(key, n, L, theta)


####################################################################################
#MCMC ABC CODE#
####################################################################################

# ─── Hyperparamètres globaux ──────────────────────────────────────────────────


# Prior : mu ~ N(0, s^2),  log(sigma^2) ~ N(0, t^2)
S_PRIOR = 1.0
T_PRIOR = 1.0



def wasserstein1(y_obs_sorted, y_sim):
    """
    Calcule W1 entre y_obs (déjà trié) et y_sim (non trié).

    Args:
        y_obs_sorted : array (m,) (données observées triées)
        y_sim        : array (m,) (données simulées non triées)
    Returns:
        scalaire JAX : distance W1
    """
    y_sim_sorted = jnp.sort(y_sim)
    return jnp.mean(jnp.abs(y_obs_sorted - y_sim_sorted))


def log_prior(theta, s=S_PRIOR, t=T_PRIOR):
    """
    Log-prior pour theta = (mu, sigma).

    Args:
        theta : array (2,)
        s, t  : hyperparamètres du prior
    Returns:
        scalaire JAX
    """
    mu, sigma = theta[0], theta[1]
    lp_mu    = -0.5 * (mu / s) ** 2                    # cste ignorée car n'apparaissent pas dans leur ratio
    lp_lsig2 = -0.5 * (2.0 * jnp.log(sigma) / t) ** 2
    return lp_mu + lp_lsig2 


def simulate(key, theta, m_sim=M_SIM, l=L):
    """
    Simule m_sim observations sous le modèle paramétré par theta.

    Args:
        key   : clé JAX PRNG
        theta : array (2,) ([mu, sigma])
        m_sim : nombre d'observations simulées
        l     : nombre de log-normales par observation
    Returns:
        y_sim : array (m_sim,) — données simulées
    """
    mu, sigma = theta[0], theta[1]
    X = mu + sigma * jax.random.normal(key, shape=(m_sim, l))
    return jnp.sum(jnp.exp(X), axis=1)     # Somme sur les colonnes



# ─── Proposition (marche aléatoire sur (mu, log sigma)) ──────────────────────
# cf. make_proposal dans le RWMH du cours.

def propose(key, theta, delta=DELTA):
    """
    Propose theta' par marche aléatoire gaussienne sur (mu, log sigma).
    Proposal symétrique donc le ratio q s'annule dans h.
    """
    mu, sigma = theta[0], theta[1]
    key_mu, key_sig = jax.random.split(key)
    mu_new       = mu + delta * jax.random.normal(key_mu)
    log_sig_new  = jnp.log(sigma) + delta * jax.random.normal(key_sig)
    sigma_new    = jnp.exp(log_sig_new)
    return jnp.array([mu_new, sigma_new])


def make_mcmc_abc(y_obs_sorted, epsilon, delta, m_sim=M_SIM, l=L):
    """
    Factory (comme make_rwmh dans le cours) qui retourne la fonction
    de chaîne unique mcmc_abc_single.

    Args:
        y_obs_sorted : données observées triées (shape (m_obs,))
        epsilon      : tolérance ABC
        m_sim, l     : paramètres de simulation
    Returns:
        mcmc_abc_single : fonction (key, theta0, n_total) -> (samples, acc_rate)
    """

    def body_fun(i, state):
        samples, theta_curr, key, n_accepted, epsilon, delta = state

        # Split de la clé en 3 subkeys : proposition, simulation, acceptation
        key, key_prop, key_sim, key_acc = jax.random.split(key, 4)

        # Proposition
        theta_new = propose(key_prop, theta_curr, delta)

        # Simulation
        y_sim = simulate(key_sim, theta_new, m_sim, l)

        # Calcul de la distance
        d = wasserstein1(y_obs_sorted, y_sim)

        # Acceptation ou rejet
        eps_accept = (d <= epsilon)
        log_h = log_prior(theta_new) - log_prior(theta_curr)
        log_u = jnp.log(jax.random.uniform(key_acc))
        accept = eps_accept & (log_h > log_u)
        theta_curr = jnp.where(accept, theta_new, theta_curr)
        n_accepted = n_accepted + jnp.where(accept, 1, 0)

        # On garde en mémoire (toujours, même si rejeté. cf. Marjoram)
        samples = samples.at[i].set(theta_curr)

        return samples, theta_curr, key, n_accepted, epsilon, delta

    def mcmc_abc_single(key, theta0, epsilon, delta, n_total=N_BURN + N_ITER):
        """
        Lance une chaîne MCMC-ABC de longueur n_total depuis theta0.
        """
        samples  = jnp.zeros((n_total, 2))
        samples  = samples.at[0].set(theta0)
        n_acc    = jnp.array(0)

        samples, _, _, n_acc, _, _ = jax.lax.fori_loop(
            1, n_total, body_fun,
            (samples, theta0, key, n_acc, jnp.array(epsilon), jnp.array(delta))
        )

        acc_rate = n_acc / (n_total - 1)
        return samples, acc_rate

    return mcmc_abc_single


# FONCTION TEMPO: petit ABC à l'avenir
def find_valid_init(key, y_obs_sorted, epsilon, n_tries=10_000,
                    s=S_PRIOR, t=T_PRIOR):
    """
    Cherche un theta0 tel que W1(Y_obs, Y_sim(theta0)) <= epsilon.
    Utilise une boucle Python ici (appelée une seule fois, avant jit).
    Inspiré de find_init dans le guide mcmc_abc du cours.
    """
    for _ in range(n_tries):
        key, k1, k2, k3 = jax.random.split(key, 4)
        # Tirage depuis le prior
        mu_try    = s * jax.random.normal(k1)
        log_s2_try = t * jax.random.normal(k2)
        sigma_try = jnp.exp(0.5 * log_s2_try)
        theta_try = jnp.array([mu_try, sigma_try])
        # Simulation et distance
        y_try = simulate(k3, theta_try)
        d     = wasserstein1(y_obs_sorted, y_try)
        if float(d) <= epsilon:
            return theta_try, key
    raise RuntimeError(
        f"Aucun theta valide trouvé en {n_tries} essais. "
    )


def run_all_chains(mcmc_abc_single, key, theta0, epsilon, delta, n_chains=N_CHAINS,
                   n_burn=N_BURN, n_iter=N_ITER):
    """
    Lance N_CHAINS chaînes en parallèle via vmap.

    Returns:
        chains_post : array (n_chains, n_iter, 2) — après burn-in
        acc_rates   : array (n_chains,) — taux d'acceptation par chaîne
    """
    subkeys = jax.random.split(key, n_chains)

    # vmap uniquement sur les keys, pas les autres params.
    mcmc_abc_vmap = jax.vmap(mcmc_abc_single, in_axes=(0, None, None, None, None))
    # Lancer toutes les chaînes en parallèle
    all_samples, all_acc_rates = mcmc_abc_vmap(subkeys, theta0, epsilon, delta, n_burn + n_iter)

    chains_post = all_samples[:, n_burn:, :]   # shape (n_chains, n_iter, 2)

    return chains_post, all_acc_rates


def compute_acf(x, max_lag=100): # Adapter aussi plot_result()
    """
    Calcule l'ACF empirique de la série x jusqu'au lag max_lag.

    Args:
        x       : array 1D (n,)
        max_lag : int
    Returns:
        acf : array (max_lag,)
    """

    acf = jnp.zeros(shape=(max_lag,))
    n = len(x)
    mean_x = jnp.mean(x)
    var_x = jnp.mean(x**2) - jnp.mean(x)**2
    for k in range(1, max_lag+1):
        acf = acf.at[k-1].set(jnp.mean((x[k:] - mean_x) * (x[:n-k] - mean_x)) / var_x)
    return acf



###########################################################################################################################################
#                                                                                                                                         #
# ---------------------------------------------------- ALGO "exact" ----------------------------------------------------------------------#
#                                                                                                                                         #
###########################################################################################################################################

def mh_sub_step_rwm(carry, _):
    """
    Noyau Metropolis-Hastings sur l'espace non-contraint (Random Walk Metropolis).
    """
    X_i, key, mu, sigma2, tau = carry
    key, k_idx, k_prop, k_acc = jr.split(key, 4)
    
    L = X_i.shape[0]
    
    # -----------------------------------------------------------------
    # 1. SÉLECTION DE LA PAIRE
    # -----------------------------------------------------------------
    idx = jr.choice(k_idx, L, shape=(2,), replace=False)
    l, lp = idx[0], idx[1]
    
    X_l = X_i[l]
    X_lp = X_i[lp]
    
    # -----------------------------------------------------------------
    # 2. CHANGEMENT DE CARTE & INVARIANTS NUMÉRIQUES
    # -----------------------------------------------------------------
    # Invariant de contrainte : log(s) = log(exp(X_l) + exp(X_lp))
    # L'utilisation de logsumexp prévient l'overflow si X_l, X_lp >> 0
    log_s = logsumexp(jnp.array([X_l, X_lp]))
    
    # État courant dans l'espace non-contraint (R)
    delta_curr = X_l - X_lp
    
    # -----------------------------------------------------------------
    # 3. PROPOSITION (RANDOM WALK)
    # -----------------------------------------------------------------
    # On propose un pas gaussien paramétré par tau. 
    # Plus besoin de borner avec un epsilon. L'espace est infini.
    delta_prop = delta_curr + tau * jr.normal(k_prop)
    
    # -----------------------------------------------------------------
    # 4. PROJECTION INVERSE VERS L'ESPACE NATIF (X)
    # -----------------------------------------------------------------
    # On reconstruit les composantes à partir de l'invariant log_s et du delta proposé.
    # softplus(x) = log(1 + exp(x)) est numériquement stable pour tout réel.
    X_l_new = log_s - softplus(-delta_prop)
    X_lp_new = log_s - softplus(delta_prop)
    
    # -----------------------------------------------------------------
    # 5. RATIO DE METROPOLIS-HASTINGS
    # -----------------------------------------------------------------
    # Miracle géométrique : le ratio de proposition q(x|y)/q(y|x) est de 1 
    # (marche aléatoire symétrique), et le Jacobien s'est annulé avec 
    # les termes de contrainte 1/u(s-u).
    # Il ne reste QUE la différence stricte des log-vraisemblances gaussiennes.
    
    log_prior_new = - ((X_l_new - mu)**2 + (X_lp_new - mu)**2) / (2.0 * sigma2)
    log_prior_curr = - ((X_l - mu)**2 + (X_lp - mu)**2) / (2.0 * sigma2)
    
    log_alpha = log_prior_new - log_prior_curr
    
    # -----------------------------------------------------------------
    # 6. ACCEPTATION
    # -----------------------------------------------------------------
    accept = jnp.log(jr.uniform(k_acc)) < log_alpha
    
    # Mise à jour conditionnelle (vectorisable par XLA)
    X_i_new = X_i.at[l].set(X_l_new).at[lp].set(X_lp_new)
    X_i_next = jnp.where(accept, X_i_new, X_i)
    
    return (X_i_next, key, mu, sigma2, tau), accept

def update_X_i_rwm(key, X_i, mu, sigma2, tau, K):
    """
    Exécute K sous-itérations du noyau Random Walk pour lisser l'autocorrélation.
    """
    init_carry = (X_i, key, mu, sigma2, tau)
    (X_i_final, _, _, _, _), _ = jax.lax.scan(mh_sub_step_rwm, init_carry, jnp.arange(K))
    return X_i_final

# Prêt pour jax.vmap sur la dimension 'batch' (n observations)
vmap_update_X_rwm = jax.vmap(update_X_i_rwm, in_axes=(0, 0, None, None, None, None))

# =====================================================================
# SCHÉMA GIBBS GLOBAL
# =====================================================================
def make_gibbs_kernel(kappa_0, m_0, a_0, b_0, tau, K):
    """Closure pour injecter les hyperparamètres (dont tau) dans le step MCMC."""
    
    def gibbs_step(carry, _):
        X, mu, sigma2, key = carry
        k_sig, k_mu, k_x, key = jr.split(key, 4)
        
        n, L = X.shape
        N = n * L
        
        # ---------------------------------------------------
        # BLOC 1 : Mise à jour exacte de (mu, sigma2 | X)
        # ---------------------------------------------------
        X_mean = jnp.mean(X)
        S2 = jnp.sum((X - X_mean)**2)
        
        kappa_n = kappa_0 + N
        m_n = (kappa_0 * m_0 + N * X_mean) / kappa_n
        a_n = a_0 + N / 2.0
        b_n = b_0 + S2 / 2.0 + (N * kappa_0) / (2.0 * kappa_n) * (X_mean - m_0)**2
        
        # Inverse-Gamma(a, b) via Gamma(a, 1) -> b / Gamma
        gamma_sample = jr.gamma(k_sig, a_n)
        sigma2_next = b_n / gamma_sample
        
        # Normal
        mu_next = m_n + jnp.sqrt(sigma2_next / kappa_n) * jr.normal(k_mu)
        
        # ---------------------------------------------------
        # BLOC 2 : Mise à jour MH de (X | mu, sigma2, Y)
        # ---------------------------------------------------
        keys_x = jr.split(k_x, n)
        
        # CRITIQUE : Utilisation du nouveau noyau vectorisé RWM avec tau
        X_next = vmap_update_X_rwm(keys_x, X, mu_next, sigma2_next, tau, K)
        
        return (X_next, mu_next, sigma2_next, key), (mu_next, sigma2_next)
    
    return gibbs_step
from functools import partial

# On indique à JAX que L, num_samples et K définissent la structure du graphe
@partial(jax.jit, static_argnames=['L', 'num_samples', 'K'])
def run_mcmc(key, Y, L, num_samples, kappa_0, m_0, a_0, b_0, K=10):
    n = Y.shape[0]
    k_init, k_scan = jr.split(key)
    
    # L est maintenant un vrai entier Python à la compilation, jnp.ones fonctionnera.
    X_init = jnp.log(Y / L)[:, None] * jnp.ones((n, L))
    mu_init = jnp.mean(X_init)
    sigma2_init = jnp.var(X_init) + 1e-3
    
    init_carry = (X_init, mu_init, sigma2_init, k_scan)
    
    # Injection du paramètre tau pour la RWM (si tu as intégré la V2)
    tau = 0.5 # A calibrer JE L AI PAS ENCORE FAIT J AI AUCUNE IDEE DE LA VALEUR OPTIMALE DE TAU NI DE COMMENT LA TROUVER
    gibbs_step = make_gibbs_kernel(kappa_0, m_0, a_0, b_0, tau, K)
    
    # num_samples est statique, donc jnp.arange(num_samples) est valide pour le scan
    _, (mu_chain, sigma2_chain) = jax.lax.scan(gibbs_step, init_carry, jnp.arange(num_samples))
    
    return mu_chain, sigma2_chain

def get_empirical_hyperparams(Y, L):
    """
    Calibre les hyperparamètres du prior Normal-Inverse-Gamma 
    à partir des statistiques des observations Y.
    """
    # Approximation : l'observation Y est répartie équitablement sur les L composantes
    log_Y_L = jnp.log(Y / L)
    
    # 1. Prior sur l'espérance mu
    m_0 = jnp.mean(log_Y_L)
    kappa_0 = 1.0  # Poids faible (équivalent à 1 observation virtuelle)
    
    # 2. Prior sur la variance sigma^2
    a_0 = 2.0      # Plus petit entier permettant d'avoir E[sigma^2] définie
    b_0 = jnp.var(log_Y_L)
    
    # Sécurité XLA : Évite un b_0 nul si Y est constant
    b_0 = jnp.maximum(b_0, 1e-3)
    
    return kappa_0, m_0, a_0, b_0

################################################################
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
################################################################



import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pandas as pd
import time
import os
from functools import partial


# ══════════════════════════════════════════════════════════════════════════════
# ABC — VERSION DYNAMIQUE
# y_obs_sorted est un argument tracé (pas une fermeture) → compile UNE SEULE
# fois pour tous les datasets de même taille.
# ══════════════════════════════════════════════════════════════════════════════

def make_mcmc_abc_dynamic(m_sim=M_SIM, l=L):
    """
    Identique à make_mcmc_abc, mais y_obs_sorted circule dans l'état
    plutôt que d'être capturé dans la fermeture.
    Avantage : JAX trace sur la forme (M_OBS,), pas sur la valeur
    → pas de recompilation entre datasets.

    Signature de mcmc_abc_single :
        (key, theta0, epsilon, delta, y_obs_sorted, n_total) -> (samples, acc_rate)
    n_total doit être statique (static_argnums dans le JIT).
    """

    def body_fun(i, state):
        samples, theta_curr, key, n_accepted, epsilon, delta, y_obs_sorted = state

        key, key_prop, key_sim, key_acc = jax.random.split(key, 4)

        theta_new = propose(key_prop, theta_curr, delta)
        y_sim     = simulate(key_sim, theta_new, m_sim, l)
        d         = wasserstein1(y_obs_sorted, y_sim)

        eps_accept = (d <= epsilon)
        log_h      = log_prior(theta_new) - log_prior(theta_curr)
        log_u      = jnp.log(jax.random.uniform(key_acc))
        accept     = eps_accept & (log_h > log_u)

        theta_curr  = jnp.where(accept, theta_new, theta_curr)
        n_accepted  = n_accepted + jnp.where(accept, 1, 0)
        samples     = samples.at[i].set(theta_curr)

        return samples, theta_curr, key, n_accepted, epsilon, delta, y_obs_sorted

    def mcmc_abc_single(key, theta0, epsilon, delta, y_obs_sorted, n_total):
        samples = jnp.zeros((n_total, 2))
        samples = samples.at[0].set(theta0)
        n_acc   = jnp.array(0)

        init_state = (
            samples, theta0, key, n_acc,
            jnp.array(epsilon), jnp.array(delta), y_obs_sorted
        )
        samples, _, _, n_acc, _, _, _ = jax.lax.fori_loop(
            1, n_total, body_fun, init_state
        )
        acc_rate = n_acc / (n_total - 1)
        return samples, acc_rate

    return mcmc_abc_single


def build_abc_runner():
    """
    Compile UNE SEULE FOIS le kernel ABC vmappé.

    Signature de la fonction retournée :
        run_abc(subkeys, theta0, epsilon, delta, y_obs_sorted, n_total)
        -> (all_samples, acc_rates)
           all_samples : (N_CHAINS, n_total, 2)
           acc_rates   : (N_CHAINS,)

    n_total est statique (static_argnums=5).
    vmap sur subkeys uniquement (in_axes=(0, None, None, None, None, None)).
    """
    mcmc_abc_single = make_mcmc_abc_dynamic()
    mcmc_abc_vmap   = jax.vmap(mcmc_abc_single,
                               in_axes=(0, None, None, None, None, None))
    mcmc_abc_jit    = jax.jit(mcmc_abc_vmap, static_argnums=(5,))
    return mcmc_abc_jit


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def section(title: str):
    width = 64
    print(f"\n{'═'*width}\n  {title}\n{'═'*width}")


def posterior_stats(samples: np.ndarray) -> dict:
    """Mean, std, median, IC95 d'un tableau 1D numpy."""
    return dict(
        mean   = float(np.mean(samples)),
        std    = float(np.std(samples)),
        median = float(np.median(samples)),
        ci_lo  = float(np.percentile(samples, 2.5)),
        ci_hi  = float(np.percentile(samples, 97.5)),
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    master_key = jr.PRNGKey(SEED)

    # ── Génération de toutes les clés de datasets à l'avance ─────────────────
    dataset_keys = jr.split(master_key, N_DATASETS + 1)
    master_key   = dataset_keys[0]
    dataset_keys = dataset_keys[1:]   # shape (N_DATASETS, 2)

    # ── Compilation JIT (une seule fois) ─────────────────────────────────────
    section("Compilation JIT")

    # ABC — warm-up avec un dataset synthétique de même forme
    print("  ABC vmap+jit... ", end="", flush=True)
    run_abc  = build_abc_runner()
    n_total  = N_BURN_ABC + N_ITER_ABC
    _Y_dummy = create_Y(jr.PRNGKey(0), M_OBS, L, (TRUE_MU, TRUE_SIG2))
    _Ys_dummy = jnp.sort(_Y_dummy)
    _th_dummy = jnp.array([TRUE_MU, TRUE_SIGMA])
    _sk_dummy = jr.split(jr.PRNGKey(1), N_CHAINS)
    t0 = time.perf_counter()
    _ = run_abc(
        _sk_dummy, _th_dummy,
        jnp.array(EPSILON_GRID[0]), jnp.array(DELTA),
        _Ys_dummy, n_total
    )[0].block_until_ready()
    print(f"ok ({time.perf_counter()-t0:.1f}s)")

    # Gibbs — warm-up
    print("  Gibbs jit...    ", end="", flush=True)
    _kp, _ma, _a0, _b0 = get_empirical_hyperparams(_Y_dummy, L)
    t0 = time.perf_counter()
    _ = run_mcmc(jr.PRNGKey(2), _Y_dummy, L, N_ITER_GIBBS,
                 _kp, _ma, _a0, _b0, K=K_GIBBS)[0].block_until_ready()
    print(f"ok ({time.perf_counter()-t0:.1f}s)")

    # ── Structures de collecte ────────────────────────────────────────────────
    rows_gibbs   = []   # posteriors_gibbs.csv
    rows_abc     = []   # posteriors_abc.csv
    rows_summary = []   # summary_per_dataset.csv

    # ── Boucle principale sur les datasets ───────────────────────────────────
    for ds_id in range(N_DATASETS):
        section(f"Dataset {ds_id+1}/{N_DATASETS}")
        key = dataset_keys[ds_id]

        # ── Créer le dataset ──────────────────────────────────────────────────
        key, k_data = jr.split(key)
        Y_obs        = create_Y(k_data, M_OBS, L, (TRUE_MU, TRUE_SIG2))
        Y_obs_sorted = jnp.sort(Y_obs)
        print(f"  Y mean={float(Y_obs.mean()):.3f}  std={float(Y_obs.std()):.3f}")

        # ── Gibbs ─────────────────────────────────────────────────────────────
        kappa_0, m_0, a_0, b_0 = get_empirical_hyperparams(Y_obs, L)
        key, k_gibbs = jr.split(key)
        t0 = time.perf_counter()
        mu_g, sig2_g = run_mcmc(k_gibbs, Y_obs, L, N_ITER_GIBBS,
                                 kappa_0, m_0, a_0, b_0, K=K_GIBBS)
        jax.block_until_ready((mu_g, sig2_g))
        t_gibbs = time.perf_counter() - t0

        mu_g_post   = np.array(mu_g[N_BURN_GIBBS:])
        sig2_g_post = np.array(sig2_g[N_BURN_GIBBS:])
        n_g         = len(mu_g_post)

        st_g = {
            "mu":   posterior_stats(mu_g_post),
            "sig2": posterior_stats(sig2_g_post),
        }

        print(f"  Gibbs  t={t_gibbs:.2f}s  "
              f"E[mu]={st_g['mu']['mean']:+.4f}  "
              f"E[sig2]={st_g['sig2']['mean']:+.4f}")

        # Stocker les échantillons Gibbs
        rows_gibbs.append(pd.DataFrame({
            "dataset_id": ds_id,
            "sample_idx": np.arange(n_g),
            "mu":         mu_g_post,
            "sig2":       sig2_g_post,
        }))

        # Résumé Gibbs
        rows_summary.append({
            "dataset_id": ds_id,
            "method":     "gibbs",
            "epsilon":    np.nan,
            "E_mu":       st_g["mu"]["mean"],
            "std_mu":     st_g["mu"]["std"],
            "E_sig2":     st_g["sig2"]["mean"],
            "std_sig2":   st_g["sig2"]["std"],
            "acc_rate":   np.nan,
            "time_s":     t_gibbs,
        })

        # ── ABC sur la grille d'epsilon ───────────────────────────────────────
        for eps in EPSILON_GRID:
            print(f"  ABC eps={eps:<5}", end="  ", flush=True)

            # Point de départ valide
            key, k_init = jr.split(key)
            try:
                theta0, key = find_valid_init(key, Y_obs_sorted, eps)
            except RuntimeError as e:
                print(f"[IGNORÉ] {e}")
                continue

            key, k_run = jr.split(key)
            subkeys = jr.split(k_run, N_CHAINS)

            t0 = time.perf_counter()
            all_samples, all_acc_rates = run_abc(
                subkeys, theta0,
                jnp.array(eps), jnp.array(DELTA),
                Y_obs_sorted, n_total
            )
            jax.block_until_ready(all_samples)
            t_abc = time.perf_counter() - t0

            # Post-traitement : burn-in + thinning
            # all_samples : (N_CHAINS, n_total, 2)
            chains_post = np.array(all_samples[:, N_BURN_ABC::K_THIN, :])
            # shape : (N_CHAINS, n_post, 2)
            n_post = chains_post.shape[1]

            acc_rates = np.array(all_acc_rates)
            mean_acc  = float(acc_rates.mean())

            # Pooler toutes les chaînes
            pooled   = chains_post.reshape(-1, 2)
            mu_abc   = pooled[:, 0]
            sig2_abc = pooled[:, 1] ** 2   # sigma → sigma2 pour comparaison

            st_abc = {
                "mu":   posterior_stats(mu_abc),
                "sig2": posterior_stats(sig2_abc),
            }

            print(f"t={t_abc:.2f}s  acc={mean_acc:.3f}  "
                  f"E[mu]={st_abc['mu']['mean']:+.4f}  "
                  f"E[sig2]={st_abc['sig2']['mean']:+.4f}")

            # Stocker les échantillons ABC (toutes chaînes, label chain_id)
            for c in range(N_CHAINS):
                chain_mu   = chains_post[c, :, 0]
                chain_sig2 = chains_post[c, :, 1] ** 2
                rows_abc.append(pd.DataFrame({
                    "dataset_id": ds_id,
                    "epsilon":    eps,
                    "chain_id":   c,
                    "sample_idx": np.arange(n_post),
                    "mu":         chain_mu,
                    "sig2":       chain_sig2,
                }))

            # Résumé ABC
            rows_summary.append({
                "dataset_id": ds_id,
                "method":     "abc",
                "epsilon":    eps,
                "E_mu":       st_abc["mu"]["mean"],
                "std_mu":     st_abc["mu"]["std"],
                "E_sig2":     st_abc["sig2"]["mean"],
                "std_sig2":   st_abc["sig2"]["std"],
                "acc_rate":   mean_acc,
                "time_s":     t_abc,
            })

    # ══════════════════════════════════════════════════════════════════════════
    # CONSTRUCTION DES DATAFRAMES FINAUX
    # ══════════════════════════════════════════════════════════════════════════
    section("Assemblage et sauvegarde")

    df_gibbs   = pd.concat(rows_gibbs,   ignore_index=True)
    df_abc     = pd.concat(rows_abc,     ignore_index=True)
    df_summary = pd.DataFrame(rows_summary)

    # ── Calcul du biais par (dataset, epsilon) ────────────────────────────────
    # Référence Gibbs par dataset
    gibbs_ref = (
        df_summary[df_summary["method"] == "gibbs"]
        [["dataset_id", "E_mu", "E_sig2"]]
        .rename(columns={"E_mu": "gibbs_E_mu", "E_sig2": "gibbs_E_sig2"})
    )

    df_abc_smry = df_summary[df_summary["method"] == "abc"].copy()
    df_abc_smry = df_abc_smry.merge(gibbs_ref, on="dataset_id")

    df_abc_smry["bias_mu"]   = (df_abc_smry["E_mu"]   - df_abc_smry["gibbs_E_mu"]).abs()
    df_abc_smry["bias_sig2"] = (df_abc_smry["E_sig2"] - df_abc_smry["gibbs_E_sig2"]).abs()

    # ── Biais moyen sur les datasets ──────────────────────────────────────────
    df_bias = (
        df_abc_smry
        .groupby("epsilon", sort=True)
        .agg(
            mean_bias_mu   = ("bias_mu",   "mean"),
            std_bias_mu    = ("bias_mu",   "std"),
            mean_bias_sig2 = ("bias_sig2", "mean"),
            std_bias_sig2  = ("bias_sig2", "std"),
            mean_acc_rate  = ("acc_rate",  "mean"),
        )
        .reset_index()
    )

    # ── Sauvegarde ────────────────────────────────────────────────────────────
    paths = {
        "posteriors_gibbs.csv":     df_gibbs,
        "posteriors_abc.csv":       df_abc,
        "summary_per_dataset.csv":  df_abc_smry,   # inclut gibbs_ref et biais par dataset
        "bias_vs_epsilon.csv":      df_bias,
    }

    for fname, df in paths.items():
        fpath = os.path.join(OUTPUT_DIR, fname)
        df.to_csv(fpath, index=False, float_format="%.8f")
        print(f"  {fname:<35s}  {len(df):>7d} lignes  →  {fpath}")

    # ── Table biais finale ────────────────────────────────────────────────────
    section("Biais moyen |E_ABC - E_Gibbs| par epsilon")
    print(df_bias.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    print("\nDone ✓")