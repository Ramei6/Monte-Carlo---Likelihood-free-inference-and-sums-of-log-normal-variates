import jax
import jax.numpy as jnp
import jax.random as jr
from jax.nn import softplus
from jax.scipy.special import logsumexp
import numpy as np
import matplotlib.pyplot as plt

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
###########################################################################################################################################
#                                                                                                                                         #
# ---------------------------------------------------- EXÉCUTION & ANALYSE DES RÉSULTATS -------------------------------------------------#
#                                                                                                                                         #
###########################################################################################################################################


@partial(jax.jit, static_argnames=['n', 'L', 'n_iter', 'burn', 'n_runs', 'K'])
def run_across_datasets(key, theta=(0.0, 0.3), n=10000, L=10, n_iter=1000, burn=200, n_runs=100, K=10):
    keys = jr.split(key, n_runs)
    
    def single_run(k):
        k_data, k_mcmc = jr.split(k)
        
        # Création et calibration
        current_Y = create_Y(k_data, n, L, theta)
        kappa_0, m_0, a_0, b_0 = get_empirical_hyperparams(current_Y, L)
        
        # MCMC
        serie_mu, serie_sigma2 = run_mcmc(
            key=k_mcmc, Y=current_Y, L=L, num_samples=n_iter, 
            kappa_0=kappa_0, m_0=m_0, a_0=a_0, b_0=b_0, K=K
        )
        
        # Troncature du burn-in
        mu_post_burn = serie_mu[burn:]
        sig2_post_burn = serie_sigma2[burn:]
        
        # Calcul des médianes
        mu_med = jnp.median(mu_post_burn)
        sig2_med = jnp.median(sig2_post_burn)
        
        # Calcul des Intervalles de Crédibilité (2.5%, 97.5%)
        # jnp.percentile prend un tableau de quantiles
        mu_ci = jnp.percentile(mu_post_burn, jnp.array([2.5, 97.5]))
        sig2_ci = jnp.percentile(sig2_post_burn, jnp.array([2.5, 97.5]))
        
        # On renvoie tout proprement séparé pour faciliter la vectorisation
        return mu_med, mu_ci[0], mu_ci[1], sig2_med, sig2_ci[0], sig2_ci[1]

    # vmap va regrouper chaque élément retourné dans un array de taille (n_runs,)
    return jax.vmap(single_run)(keys)
###########################################################################################################################################

import os
import time

if __name__ == "__main__": 
    # Tes paramètres vrais pour vérifier le biais
    theta_true = (0.0, 0.09)
    n_runs_target = 100
    
    print("🔥 Lancement du Warm-up (Compilation XLA)...")
    t0_warmup = time.time()
    
    # WARM-UP (on ignore les sorties)
    _ = run_across_datasets(
        jr.PRNGKey(99), theta=theta_true, n=10000, L=10, 
        n_iter=1000, burn=200, n_runs=n_runs_target, K=10
    )[0].block_until_ready()
    
    t1_warmup = time.time()
    print(f"⏱️ Temps de compilation : {t1_warmup - t0_warmup:.2f} s\n")

    print(f"🚀 Lancement de {n_runs_target} Runs MCMC (benchmark pur)...")
    t0_run = time.time()
    
    # RUN RÉEL : On récupère les 6 tenseurs générés par notre nouvelle fonction
    results = run_across_datasets(
        jr.PRNGKey(0), theta=theta_true, n=10000, L=10, 
        n_iter=1000, burn=200, n_runs=n_runs_target, K=10
    )
    
    # SYNCHRONISATION : On s'assure que tout est calculé
    results = jax.tree_util.tree_map(lambda x: x.block_until_ready(), results)
    t1_run = time.time()
    
    print(f"⏱️ Temps d'exécution pur : {t1_run - t0_run:.2f} s")
    print(f"⏱️ Temps moyen par dataset : {(t1_run - t0_run) / n_runs_target:.3f} s\n")
    
    # Déballage et conversion en NumPy
    mu_meds, mu_lows, mu_highs, sig2_meds, sig2_lows, sig2_highs = [np.array(x) for x in results]

    # --- RÉSULTATS GLOBAUX ---
    # Médiane des médianes
    global_mu = np.median(mu_meds)
    global_sig2 = np.median(sig2_meds)

    # -------------------------------------------------------------------------
    # SAUVEGARDE ET PLOTS
    # -------------------------------------------------------------------------
    output_dir = "mcmc_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Sauvegarde Data
    np.savez_compressed(
        os.path.join(output_dir, "mcmc_results_100runs_sigma2=0.09.npz"), 
        mu_meds=mu_meds, mu_lows=mu_lows, mu_highs=mu_highs,
        sig2_meds=sig2_meds, sig2_lows=sig2_lows, sig2_highs=sig2_highs
    )

    # --- CRÉATION DE LA FIGURE 2x2 ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    x_runs = np.arange(n_runs_target)

    # 1. Trace de Mu avec IC
    axes[0, 0].fill_between(x_runs, mu_lows, mu_highs, color='blue', alpha=0.2, label="IC 95%")
    axes[0, 0].plot(x_runs, mu_meds, color='blue', marker='.', markersize=4, linestyle='-', label="Médiane estimée")
    axes[0, 0].axhline(theta_true[0], color='black', linestyle='--', linewidth=2, label="Vraie valeur")
    axes[0, 0].set_title(r"Trajectoire des estimations de $\mu$ par Run")
    axes[0, 0].set_xlabel("Run MCMC")
    axes[0, 0].legend()

    # 2. Trace de Sigma^2 avec IC
    axes[0, 1].fill_between(x_runs, sig2_lows, sig2_highs, color='green', alpha=0.2, label="IC 95%")
    axes[0, 1].plot(x_runs, sig2_meds, color='green', marker='.', markersize=4, linestyle='-', label="Médiane estimée")
    axes[0, 1].axhline(theta_true[1], color='black', linestyle='--', linewidth=2, label="Vraie valeur")
    axes[0, 1].set_title(r"Trajectoire des estimations de $\sigma^2$ par Run")
    axes[0, 1].set_xlabel("Run MCMC")
    axes[0, 1].legend()

    # 3. Histogramme des médianes de Mu
    axes[1, 0].hist(mu_meds, bins=15, color='blue', alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(theta_true[0], color='black', linestyle='--', linewidth=2, label="Vraie valeur")
    axes[1, 0].axvline(global_mu, color='red', linestyle=':', linewidth=2, label="Médiane empirique")
    axes[1, 0].set_title(r"Distribution des $100$ estimateurs de $\mu$")
    axes[1, 0].set_xlabel("Valeur de la médiane")
    axes[1, 0].legend()

    # 4. Histogramme des médianes de Sigma^2
    axes[1, 1].hist(sig2_meds, bins=15, color='green', alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(theta_true[1], color='black', linestyle='--', linewidth=2, label="Vraie valeur")
    axes[1, 1].axvline(global_sig2, color='red', linestyle=':', linewidth=2, label="Médiane empirique")
    axes[1, 1].set_title(r"Distribution des $100$ estimateurs de $\sigma^2$")
    axes[1, 1].set_xlabel("Valeur de la médiane")
    axes[1, 1].legend()

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "mcmc_100runs_sigma2=0.09.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"📊 Graphiques sauvegardés dans : {plot_path}")