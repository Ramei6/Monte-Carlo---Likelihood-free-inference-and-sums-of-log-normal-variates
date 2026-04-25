"""
MCMC-ABC (Marjoram et al., 2003)
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from functools import partial

# ─── Hyperparamètres globaux ──────────────────────────────────────────────────
L       = 10        # nombre de log-normales par observation
M_OBS   = 500       # taille du jeu de données observé
M_SIM   = 500       # taille du jeu simulé à chaque étape MH
TRUE_MU    = 0.0    # vraie valeur de mu
TRUE_SIGMA = 0.3    # vraie valeur de sigma

# Prior : mu ~ N(0, s^2),  log(sigma^2) ~ N(0, t^2)
S_PRIOR = 1.0
T_PRIOR = 1.0

# Chaîne MCMC
N_CHAINS  = 8       # nombre de chaînes parallèles (vmappées)
N_BURN    = 2_000   # longueur du burn-in
N_ITER    = 10_000  # longueur post-burn-in
K_THIN    = 5       # facteur de thinning pour les diagnostics

EPSILON   = 1.5     # tolérance ABC (à calibrer via pilot run)
DELTA_MU      = 0.10   # std de la marche aléatoire sur mu
DELTA_LOG_SIG = 0.10   # std de la marche aléatoire sur log(sigma)

# ─── 0. Génération des données "observées" ───────────────────────────────────

def generate_observed_data(key, mu, sigma, m, l):
    """Génère m observations Yi = sum_l exp(Xi,l) avec Xi,l ~ N(mu, sigma^2)."""
    X = random.normal(key, shape=(m, l))         # shape (m, L)
    return jnp.sum(jnp.exp(mu + sigma * X), axis=1)  # shape (m,)

key_master = random.PRNGKey(42)
key_obs, key_run = random.split(key_master)
Y_OBS = generate_observed_data(key_obs, TRUE_MU, TRUE_SIGMA, M_OBS, L)
Y_OBS_sorted = jnp.sort(Y_OBS)   # pré-tri pour W1 efficace


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

def propose(key, theta, delta_mu=DELTA_MU, delta_log_sig=DELTA_LOG_SIG):
    """
    Propose theta' par marche aléatoire gaussienne sur (mu, log sigma).
    Proposal symétrique donc le ratio q s'annule dans h.
    """
    mu, sigma = theta[0], theta[1]
    key_mu, key_sig = random.split(key)
    mu_new       = mu + delta_mu * jax.random.normal(key_mu)
    log_sig_new  = jnp.log(sigma) + delta_log_sig * jax.random.normal(key_sig)
    sigma_new    = jnp.exp(log_sig_new)
    return jnp.array([mu_new, sigma_new])


# ═══════════════════════════════════════════════════════════════════════════════
# BLOC D — Corps de la chaîne MCMC-ABC (une itération)  [À COMPLÉTER]
# ═══════════════════════════════════════════════════════════════════════════════
#
# C'est le cœur de l'algorithme F de Marjoram et al.
# Cette fonction sera passée à lax.fori_loop — signature obligatoire :
#   body_fun(i, state) -> state
#
# L'état de la chaîne est un tuple :
#   state = (samples, theta_curr, key, n_accepted)
#
# À chaque itération i :
#   F1. Proposer theta_new = propose(key_prop, theta_curr)
#   F2. Simuler Y_sim ~ modèle(theta_new)
#   F3'. Si W1(Y_obs_sorted, Y_sim) > epsilon → rester, enregistrer theta_curr
#   F4. log_h = log_prior(theta_new) - log_prior(theta_curr)
#        (le ratio q s'annule car proposal symétrique)
#   F5. u ~ Uniform(0,1); si log(u) < log_h → accepter theta_new
#   Toujours : samples = samples.at[i].set(theta_curr après décision)
#
# POINTS CRITIQUES JAX :
#   - Pas de if/else sur des valeurs JAX → utilise jnp.where(cond, a, b)
#     pour choisir entre theta_new et theta_curr
#   - La porte F3' ET la condition MH F5 peuvent être combinées :
#       accept = within_eps & (log_u < log_h)
#   - n_accepted doit être mis à jour avec jnp.where aussi
#   - Utilise arr.at[i].set(val) pour écrire dans samples (immutabilité)

def make_mcmc_abc(y_obs_sorted, epsilon, m_sim=M_SIM, l=L):
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
        samples, theta_curr, key, n_accepted = state

        # Split de la clé en 3 subkeys : proposition, simulation, acceptation
        key, key_prop, key_sim, key_acc = jax.random.split(key, 4)

        # Proposition
        theta_new = propose(key_prop, theta_curr)

        # Simulation
        y_sim = simulate(key_sim, theta_new, m_sim, l)

        # Calcul de la distance
        d = wasserstein1(y_obs_sorted, y_sim)

        # Acceptation ou rejet
        eps_accept = (d <= epsilon)
        log_h = log_prior(theta_new) - log_prior(theta_curr)
        log_u = jnp.log(jax.random.uniform(key_acc))
        theta_curr = jnp.where(eps_accept, theta_new, theta_curr)
        n_accepted = n_accepted + jnp.where(eps_accept, 1, 0)

        # On garde en mémoire (toujours, même si rejeté. cf. Marjoram)
        samples = samples.at[i].set(theta_curr)

        return samples, theta_curr, key, n_accepted

    def mcmc_abc_single(key, theta0, n_total=N_BURN + N_ITER):
        """
        Lance une chaîne MCMC-ABC de longueur n_total depuis theta0.
        """
        samples  = jnp.zeros((n_total, 2))
        samples  = samples.at[0].set(theta0)
        n_acc    = jnp.array(0)

        samples, _, _, n_acc = jax.lax.fori_loop(
            1, n_total, body_fun,
            (samples, theta0, key, n_acc)
        )

        acc_rate = n_acc / (n_total - 1)
        return samples, acc_rate

    return mcmc_abc_single


# FONCTION TEMPO: petit ABC à l'avenir
def find_valid_init(key, y_obs_sorted, epsilon, n_tries=5_000,
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


# ═══════════════════════════════════════════════════════════════════════════════
# BLOC E — Vectorisation sur N_CHAINS avec vmap  [À COMPLÉTER]
# ═══════════════════════════════════════════════════════════════════════════════
#
# Reproduis exactement le pattern du RWMH du cours :
#
#   rwmh_vmap = vmap(rwmh, in_axes=(0, None, None, None))
#   keys = random.split(PRNGKey(0), n_chains)
#   samples = rwmh_vmap(keys, x0, n_samples, burnin)
#
# Ici :
#   mcmc_abc_vmap = vmap(mcmc_abc_single, in_axes=(0, None, None))
#   keys = random.split(key_run, N_CHAINS)
#   all_samples, all_acc_rates = mcmc_abc_vmap(keys, theta0, N_BURN + N_ITER)
#
# IMPORTANT :
#   - theta0 est le MÊME pour toutes les chaînes (in_axes=None dessus)
#   - Seules les clés sont différentes (in_axes=0)
#   - Résultats : all_samples.shape = (N_CHAINS, N_BURN+N_ITER, 2)

def run_all_chains(mcmc_abc_single, key, theta0, n_chains=N_CHAINS,
                   n_burn=N_BURN, n_iter=N_ITER):
    """
    Lance N_CHAINS chaînes en parallèle via vmap.

    Returns:
        chains_post : array (n_chains, n_iter, 2) — après burn-in
        acc_rates   : array (n_chains,) — taux d'acceptation par chaîne
    """
    subkeys = jax.random.split(key, n_chains)

    # vmap uniquement sur les keys, pas les autres params.
    mcmc_abc_vmap = jax.vmap(mcmc_abc_single, in_axis=(0, None, None))
    # Lancer toutes les chaînes en parallèle
    all_samples, all_acc_rates = mcmc_abc_vmap(subkeys, theta0, n_burn + n_iter)

    chains_post = all_samples[:, n_burn:, :]   # shape (n_chains, n_iter, 2)

    return chains_post, all_acc_rates


# ═══════════════════════════════════════════════════════════════════════════════
# BLOC F — Diagnostics  [À COMPLÉTER]
# ═══════════════════════════════════════════════════════════════════════════════
#
# Implémente l'ACF (autocorrélation empirique) pour une seule série.
# Formule : pour le lag k,
#   ACF(k) = mean((x[k:] - mean(x)) * (x[:-k] - mean(x))) / var(x)
#
# HINT : même logique que la fonction acf() dans le notebook RWMH du cours,
#        mais vectorisée avec jnp (pas de boucle Python sur k).
#        Tu peux utiliser une boucle Python sur les lags (petit nombre),
#        mais les opérations internes doivent être JAX.

def compute_acf(x, max_lag=50):
    """
    Calcule l'ACF empirique de la série x jusqu'au lag max_lag.

    Args:
        x       : array 1D (n,)
        max_lag : int
    Returns:
        acf : array (max_lag,)
    """
    # ── TON CODE ICI ──────────────────────────────────────────────────────────
    # n = len(x)
    # mean_x = jnp.mean(x)
    # var_x  = jnp.mean((x - mean_x)**2)
    # Pour k in range(1, max_lag+1) :
    #   acf[k-1] = mean((x[k:] - mean_x) * (x[:n-k] - mean_x)) / var_x
    raise NotImplementedError("BLOC F — À compléter")
    # ─────────────────────────────────────────────────────────────────────────


# ─── Visualisation des résultats ─────────────────────────────────────────────

def plot_results(chains_post, acc_rates, epsilon, k_thin=K_THIN,
                 true_mu=TRUE_MU, true_sigma=TRUE_SIGMA):
    """
    Produit 4 figures de diagnostic :
      1. Trace plots (toutes les chaînes, mu et sigma)
      2. ACF (chaîne 0, mu et sigma)
      3. Posterior marginals : histogrammes + boxplots + IC 95%
      4. Taux d'acceptation par chaîne
    """
    n_chains, n_iter, _ = chains_post.shape

    # Thinning pour les histogrammes
    chains_thin = chains_post[:, ::k_thin, :]          # (n_chains, n_iter//k, 2)
    flat_mu    = np.array(chains_thin[:, :, 0]).ravel()
    flat_sigma = np.array(chains_thin[:, :, 1]).ravel()

    fig = plt.figure(figsize=(14, 12))
    fig.suptitle(f"MCMC-ABC — Marjoram et al. (2003)   [ε = {epsilon}]",
                 fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── 1. Trace plots ────────────────────────────────────────────────────────
    ax_tr_mu  = fig.add_subplot(gs[0, 0])
    ax_tr_sig = fig.add_subplot(gs[0, 1])
    for c in range(n_chains):
        ax_tr_mu.plot(chains_post[c, ::20, 0], lw=0.5, alpha=0.7)
        ax_tr_sig.plot(chains_post[c, ::20, 1], lw=0.5, alpha=0.7)
    for ax, name, true in [(ax_tr_mu, "μ", true_mu),
                            (ax_tr_sig, "σ", true_sigma)]:
        ax.axhline(true, color="k", lw=1.5, ls="--", label=f"Vraie valeur {true}")
        ax.set_title(f"Trace — {name}")
        ax.set_xlabel("Itération (thinning ×20)")
        ax.legend(fontsize=8)

    # ── 2. ACF ────────────────────────────────────────────────────────────────
    ax_acf_mu  = fig.add_subplot(gs[1, 0])
    ax_acf_sig = fig.add_subplot(gs[1, 1])
    lags = np.arange(1, 51)
    for c in range(min(3, n_chains)):
        acf_mu  = np.array(compute_acf(chains_post[c, :, 0]))
        acf_sig = np.array(compute_acf(chains_post[c, :, 1]))
        ax_acf_mu.plot(lags, acf_mu, lw=1, alpha=0.8, label=f"Chaîne {c}")
        ax_acf_sig.plot(lags, acf_sig, lw=1, alpha=0.8, label=f"Chaîne {c}")
    for ax, name in [(ax_acf_mu, "μ"), (ax_acf_sig, "σ")]:
        ax.axhline(0.1, color="red", ls="--", lw=1, label="Seuil 0.1")
        ax.axhline(0.0, color="k", lw=0.5)
        ax.set_title(f"ACF — {name}")
        ax.set_xlabel("Lag")
        ax.legend(fontsize=8)

    # ── 3. Posterior : histogramme + boxplot + IC 95% ─────────────────────────
    ax_post_mu  = fig.add_subplot(gs[2, 0])
    ax_post_sig = fig.add_subplot(gs[2, 1])
    for ax, samples, name, true in [
        (ax_post_mu,  flat_mu,    "μ",    true_mu),
        (ax_post_sig, flat_sigma, "σ", true_sigma),
    ]:
        ax.hist(samples, bins=60, density=True, alpha=0.6,
                color="steelblue", edgecolor="white", linewidth=0.3)
        q025, q50, q975 = np.percentile(samples, [2.5, 50, 97.5])
        ax.axvline(true,  color="black",  lw=2,   ls="--", label=f"Vraie valeur ({true})")
        ax.axvline(q50,   color="crimson",lw=1.5,           label=f"Médiane ({q50:.3f})")
        ax.axvspan(q025, q975, alpha=0.15, color="crimson",
                   label=f"IC 95% [{q025:.3f}, {q975:.3f}]")
        ax.set_title(f"Posterior marginal — {name}  (thinning ×{k_thin})")
        ax.set_xlabel(name)
        ax.legend(fontsize=7)
        # Statistiques textuelles
        ax.text(0.98, 0.95,
                f"Mean={np.mean(samples):.3f}\nStd={np.std(samples):.3f}\n"
                f"ESS≈{int(len(samples)/(1+2*float(compute_acf(jnp.array(samples[:2000]))[0])))}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8, bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    plt.savefig("/mnt/user-data/outputs/mcmc_abc_results.png",
                dpi=150, bbox_inches="tight")
    plt.show()
    print("Figure sauvegardée.")


def plot_sensitivity_epsilon(results_by_eps, true_mu=TRUE_MU, true_sigma=TRUE_SIGMA):
    """
    Boxplots des posteriors de mu et sigma pour différentes valeurs de epsilon.
    Essentiel pour la comparaison avec Reject-ABC (Question 4 du sujet).

    Args:
        results_by_eps : dict { epsilon : (flat_mu, flat_sigma) }
    """
    epsilons = sorted(results_by_eps.keys())
    data_mu    = [results_by_eps[e][0] for e in epsilons]
    data_sigma = [results_by_eps[e][1] for e in epsilons]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Sensibilité au choix de ε — MCMC-ABC", fontweight="bold")

    for ax, data, name, true in [
        (axes[0], data_mu,    "μ",    true_mu),
        (axes[1], data_sigma, "σ", true_sigma),
    ]:
        bp = ax.boxplot(data, labels=[str(e) for e in epsilons],
                        patch_artist=True, notch=True,
                        medianprops=dict(color="crimson", lw=2))
        for patch in bp["boxes"]:
            patch.set_facecolor("steelblue")
            patch.set_alpha(0.5)
        ax.axhline(true, color="k", ls="--", lw=1.5, label=f"Vraie valeur ({true})")
        ax.set_xlabel("ε")
        ax.set_title(f"Posterior de {name}")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/mcmc_abc_sensitivity_eps.png",
                dpi=150, bbox_inches="tight")
    plt.show()


# ─── Script principal ─────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=== MCMC-ABC (Marjoram et al., 2003) — JAX ===\n")
    print(f"Données observées : {M_OBS} obs, L={L}, mu={TRUE_MU}, sigma={TRUE_SIGMA}")
    print(f"ε = {EPSILON},  {N_CHAINS} chaînes,  {N_BURN} burn-in + {N_ITER} iter\n")

    # 1. Initialisation valide
    print("Recherche d'un theta0 valide...")
    theta0, key_run = find_valid_init(key_run, Y_OBS_sorted, EPSILON)
    print(f"theta0 = (mu={float(theta0[0]):.3f}, sigma={float(theta0[1]):.3f})\n")

    # 2. Construction de la chaîne
    mcmc_abc_single = make_mcmc_abc(Y_OBS_sorted, EPSILON)

    # 3. JIT + lancement (le premier appel compile — prend ~30s)
    print("Compilation et lancement des chaînes (vmap + jit)...")
    chains_post, acc_rates = run_all_chains(
        mcmc_abc_single, key_run, theta0
    )
    print(f"Taux d'acceptation par chaîne : {np.array(acc_rates)}")
    print(f"Taux moyen : {float(jnp.mean(acc_rates)):.3f}\n")

    # 4. Diagnostics visuels
    plot_results(chains_post, acc_rates, EPSILON)

    # 5. Analyse de sensibilité à epsilon
    print("\n=== Analyse de sensibilité à ε ===")
    results_by_eps = {}
    for eps in [0.5, 1.0, 1.5, 2.5]:
        print(f"  ε = {eps}...")
        # ── TON CODE ICI ──────────────────────────────────────────────────────
        # Pour chaque epsilon :
        #   1. Construis make_mcmc_abc(Y_OBS_sorted, eps)
        #   2. Lance run_all_chains
        #   3. Thinne et mets à plat les résultats
        #   4. Stocke dans results_by_eps[eps] = (flat_mu, flat_sigma)
        pass
        # ─────────────────────────────────────────────────────────────────────

    if results_by_eps:
        plot_sensitivity_epsilon(results_by_eps)