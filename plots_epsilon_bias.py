"""
plot_epsilon_bias.py
====================
Visualisation des résultats de l'expérience biais-epsilon.
Charge les CSV générés par epsilon_bias_experiment.py et produit 4 figures.

Figures :
  fig1_bias_vs_epsilon.png       — Biais moyen ± std sur les datasets
  fig2_posterior_shift.png       — Superposition des posteriors (un dataset de référence)
  fig3_bias_per_dataset.png      — Distribution du biais par dataset (boxplot)
  fig4_bias_acceptance_tradeoff  — Biais vs taux d'acceptation (axe double)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
import seaborn as sns

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

INPUT_DIR  = "epsilon_comparison"
OUTPUT_DIR = "epsilon_comparison/figures"
REF_DS     = 0          # dataset de référence pour la figure 2

TRUE_MU   = 0.0
TRUE_SIG2 = 0.09

# ── Palette ε : dégradé bleu-ciel → rouge brique ─────────────────────────────
# Sera définie dynamiquement après chargement (on ne connaît pas encore la grille)

# ── Style global ──────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", context="paper", font_scale=1.25)
plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Palatino Linotype", "Palatino", "Georgia", "DejaVu Serif"],
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.8,
    "grid.linewidth":    0.5,
    "grid.alpha":        0.4,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
})

GIBBS_COLOR  = "#2C3E50"     # bleu nuit — Gibbs / référence exacte
ACCENT_COLOR = "#C0392B"     # rouge — accentuation (biais, axe secondaire)
BAND_ALPHA   = 0.18


# ══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT
# ══════════════════════════════════════════════════════════════════════════════

def load_data(input_dir: str):
    df_bias    = pd.read_csv(os.path.join(input_dir, "bias_vs_epsilon.csv"))
    df_summary = pd.read_csv(os.path.join(input_dir, "summary_per_dataset.csv"))
    df_gibbs   = pd.read_csv(os.path.join(input_dir, "posteriors_gibbs.csv"))
    df_abc     = pd.read_csv(os.path.join(input_dir, "posteriors_abc.csv"))
    return df_bias, df_summary, df_gibbs, df_abc


def make_epsilon_palette(epsilons):
    """Dégradé du bleu clair au rouge brique sur la grille epsilon."""
    cmap   = plt.cm.get_cmap("RdYlBu_r", len(epsilons))
    return {eps: cmap(i / (len(epsilons) - 1)) for i, eps in enumerate(epsilons)}


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Biais moyen vs ε  (±1 std sur les datasets)
# ══════════════════════════════════════════════════════════════════════════════

def plot_bias_vs_epsilon(df_bias: pd.DataFrame, eps_palette: dict, out_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    fig.suptitle(
        r"ABC bias as a function of tolerance $\varepsilon$",
        fontsize=14, fontweight="bold", y=1.02
    )

    specs = [
        ("mean_bias_mu",   "std_bias_mu",   r"$\mu$",        axes[0]),
        ("mean_bias_sig2", "std_bias_sig2", r"$\sigma^2$",   axes[1]),
    ]
    epsilons = df_bias["epsilon"].values

    for mean_col, std_col, param_label, ax in specs:
        mean_vals = df_bias[mean_col].values
        std_vals  = df_bias[std_col].values

        # Shaded band ±1 std
        ax.fill_between(epsilons,
                        mean_vals - std_vals,
                        mean_vals + std_vals,
                        color=ACCENT_COLOR, alpha=BAND_ALPHA, zorder=1)

        # Mean line with markers colored by epsilon
        for i in range(len(epsilons) - 1):
            ax.plot(epsilons[i:i+2], mean_vals[i:i+2],
                    color=ACCENT_COLOR, lw=1.8, zorder=2)

        # Scatter points colored by epsilon value
        for eps, m in zip(epsilons, mean_vals):
            ax.scatter(eps, m, color=eps_palette[eps], s=70,
                       zorder=3, edgecolors="white", linewidths=0.8)

        ax.set_xlabel(r"Tolerance $\varepsilon$", labelpad=6)
        ax.set_ylabel(
            r"$\left|\,\mathbb{E}_{\mathrm{ABC}}[\theta] - \mathbb{E}_{\mathrm{Gibbs}}[\theta]\,\right|$",
            labelpad=6
        )
        ax.set_title(f"Parameter {param_label}", fontsize=12, pad=8)
        ax.set_xlim(epsilons.min() * 0.85, epsilons.max() * 1.05)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
        ax.tick_params(axis="both", labelsize=10)

        # Annotate the rightmost point
        ax.annotate(
            f"ε={epsilons[-1]:.1f}\nbias={mean_vals[-1]:.4f}",
            xy=(epsilons[-1], mean_vals[-1]),
            xytext=(-42, 12), textcoords="offset points",
            fontsize=8, color=ACCENT_COLOR,
            arrowprops=dict(arrowstyle="-", color=ACCENT_COLOR, lw=0.8),
        )

    fig.tight_layout()
    path = os.path.join(out_dir, "fig1_bias_vs_epsilon.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Dérive des posteriors avec ε  (KDE superposées, 1 dataset)
# ══════════════════════════════════════════════════════════════════════════════

def plot_posterior_shift(df_gibbs, df_abc, eps_palette, ref_ds, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        rf"Posterior shift with $\varepsilon$  —  dataset {ref_ds}",
        fontsize=14, fontweight="bold", y=1.02
    )

    gibbs_ds  = df_gibbs[df_gibbs["dataset_id"] == ref_ds]
    abc_ds    = df_abc[df_abc["dataset_id"] == ref_ds]
    epsilons  = sorted(abc_ds["epsilon"].unique())

    specs = [
        ("mu",   r"$\mu$",       TRUE_MU,   axes[0], -0.5, 0.15),
        ("sig2", r"$\sigma^2$",  TRUE_SIG2, axes[1],  0.0, 0.50),
    ]

    for param, label, true_val, ax, x_min, x_max in specs:
        xs = np.linspace(x_min, x_max, 500)
        
        # ── 1. Inset Axes Setup ───────────────────────────────────────────
        # Place inset top-left for mu, top-right for sig2 to avoid blocking data
        inset_rect = [0.05, 0.55, 0.35, 0.40] if param == "mu" else [0.60, 0.55, 0.35, 0.40]
        axins = ax.inset_axes(inset_rect)
        
        # Variable to track the highest ABC peak for main axis scaling
        max_abc_density = 0

        # ── 2. Gibbs KDE (Reference) ──────────────────────────────────────
        g_vals = gibbs_ds[param].values
        g_kde  = gaussian_kde(g_vals, bw_method="silverman")
        g_density = g_kde(xs)

        # Plot on main axis (will be truncated by ylim later)
        ax.plot(xs, g_density, color=GIBBS_COLOR, lw=2.5, ls="--", label="Gibbs (exact)", zorder=10)
        
        # Plot full spike on inset axis
        axins.plot(xs, g_density, color=GIBBS_COLOR, lw=1.5, ls="--", zorder=10)
        
        # Make the True Value line bolder (red) so it stands out
        ax.axvline(true_val, color="red", lw=1.5, ls=":", alpha=0.8, label="True value", zorder=15)
        axins.axvline(true_val, color="red", lw=1.0, ls=":", alpha=0.8, zorder=15)

        # ── 3. ABC KDE per epsilon ────────────────────────────────────────
        for eps in epsilons:
            vals = abc_ds[abc_ds["epsilon"] == eps][param].values
            if len(vals) < 20:
                continue
            kde  = gaussian_kde(vals, bw_method="silverman")
            col  = eps_palette[eps]
            density = kde(xs)
            
            # Update the max ABC density found
            max_abc_density = max(max_abc_density, density.max())

            # Plot on main axis
            ax.plot(xs, density, color=col, lw=1.5, alpha=0.85, label=rf"$\varepsilon$={eps}")
            ax.fill_between(xs, density, alpha=0.04, color=col)
            
            # Plot on inset (they will look flat, providing context)
            axins.plot(xs, density, color=col, lw=1.0, alpha=0.4)

        # ── 4. Axis Limits and Formatting ─────────────────────────────────
        ax.set_xlim(x_min, x_max)
        axins.set_xlim(x_min, x_max)
        
        # Dynamically cap the main Y-axis ~15% above the highest ABC curve
        ax.set_ylim(0, max_abc_density * 1.15)
        
        # Allow the inset to scale entirely to the Gibbs spike
        axins.set_ylim(0, g_density.max() * 1.10)
        axins.set_yticks([]) # Hide inset Y-ticks to keep it clean
        axins.set_title("Full Scale", fontsize=9, pad=3)

        ax.set_xlabel(label, labelpad=6)
        ax.set_ylabel("Density", labelpad=6)
        ax.set_title(f"Posterior of {label}", fontsize=12, pad=8)
        ax.tick_params(axis="both", labelsize=10)

    # ── Shared Legend ─────────────────────────────────────────────────────────
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="center right",
               bbox_to_anchor=(1.18, 0.5),
               fontsize=9,
               frameon=True,
               framealpha=0.9,
               edgecolor="#cccccc",
               title=r"Method / $\varepsilon$",
               title_fontsize=9)

    fig.tight_layout()
    path = os.path.join(out_dir, "fig2_posterior_shift.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Distribution du biais par dataset (boxplot + strip)
# ══════════════════════════════════════════════════════════════════════════════

def plot_bias_per_dataset(df_summary, eps_palette, out_dir):
    # df_summary contient une ligne par (dataset_id, epsilon) pour les ABC
    abc_rows = df_summary[df_summary["method"] == "abc"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        r"Distribution of bias $|\mathbb{E}_{\mathrm{ABC}} - \mathbb{E}_{\mathrm{Gibbs}}|$ across datasets",
        fontsize=14, fontweight="bold", y=1.02
    )

    specs = [
        ("bias_mu",   r"$|\,\mathbb{E}[\mu]_{\mathrm{ABC}} - \mathbb{E}[\mu]_{\mathrm{Gibbs}}\,|$",   axes[0]),
        ("bias_sig2", r"$|\,\mathbb{E}[\sigma^2]_{\mathrm{ABC}} - \mathbb{E}[\sigma^2]_{\mathrm{Gibbs}}\,|$", axes[1]),
    ]

    for bias_col, ylabel, ax in specs:
        epsilons_sorted = sorted(abc_rows["epsilon"].unique())
        palette_ordered = [eps_palette[e] for e in epsilons_sorted]

        sns.boxplot(
            data=abc_rows,
            x="epsilon", y=bias_col,
            palette=palette_ordered,
            order=epsilons_sorted,
            width=0.55,
            fliersize=0,          # on dessine les points manuellement
            linewidth=1.2,
            ax=ax,
        )
        sns.stripplot(
            data=abc_rows,
            x="epsilon", y=bias_col,
            palette=palette_ordered,
            order=epsilons_sorted,
            size=5.5,
            jitter=0.18,
            alpha=0.75,
            edgecolor="white",
            linewidth=0.5,
            ax=ax,
        )

        ax.set_xlabel(r"Tolerance $\varepsilon$", labelpad=6)
        ax.set_ylabel(ylabel, labelpad=6)
        ax.tick_params(axis="x", labelsize=9)
        ax.tick_params(axis="y", labelsize=10)

    fig.tight_layout()
    path = os.path.join(out_dir, "fig3_bias_per_dataset.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Biais vs taux d'acceptation (axe Y double)
# ══════════════════════════════════════════════════════════════════════════════

def plot_bias_acceptance_tradeoff(df_bias, eps_palette, out_dir):
    epsilons  = df_bias["epsilon"].values
    acc_rates = df_bias["mean_acc_rate"].values
    bias_mu   = df_bias["mean_bias_mu"].values
    bias_sig2 = df_bias["mean_bias_sig2"].values
    colors_eps = [eps_palette[e] for e in epsilons]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    fig.suptitle(
        r"Bias–acceptance tradeoff as $\varepsilon$ grows",
        fontsize=14, fontweight="bold", y=1.02
    )

    ax2 = ax1.twinx()

    # ── Bias lines (left axis) ────────────────────────────────────────────────
    ax1.plot(epsilons, bias_mu,   color=ACCENT_COLOR, lw=2.0,
             ls="-",  label=r"Bias $\mu$",      zorder=3)
    ax1.plot(epsilons, bias_sig2, color=ACCENT_COLOR, lw=2.0,
             ls="--", label=r"Bias $\sigma^2$", zorder=3)

    # Scatter points colored by ε
    for eps, bm, bs in zip(epsilons, bias_mu, bias_sig2):
        c = eps_palette[eps]
        ax1.scatter(eps, bm, color=c, s=65, zorder=5,
                    edgecolors="white", linewidths=0.8)
        ax1.scatter(eps, bs, color=c, s=65, zorder=5,
                    edgecolors="white", linewidths=0.8, marker="D")

    ax1.set_xlabel(r"Tolerance $\varepsilon$", labelpad=6)
    ax1.set_ylabel("Mean absolute bias", color=ACCENT_COLOR, labelpad=6)
    ax1.tick_params(axis="y", labelcolor=ACCENT_COLOR, labelsize=10)
    ax1.tick_params(axis="x", labelsize=10)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))

    # ── Acceptance rate (right axis) ─────────────────────────────────────────
    ax2.plot(epsilons, acc_rates, color=GIBBS_COLOR, lw=2.0,
             ls="-.", label="Acceptance rate", zorder=2)
    ax2.scatter(epsilons, acc_rates, color=GIBBS_COLOR, s=55,
                zorder=4, edgecolors="white", linewidths=0.8)

    ax2.set_ylabel("Mean acceptance rate", color=GIBBS_COLOR, labelpad=6)
    ax2.tick_params(axis="y", labelcolor=GIBBS_COLOR, labelsize=10)
    ax2.set_ylim(0, min(1.05, acc_rates.max() * 2.0))
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    # ── Colorbar to show ε mapping ────────────────────────────────────────────
    sm = plt.cm.ScalarMappable(
        cmap="RdYlBu_r",
        norm=plt.Normalize(vmin=epsilons.min(), vmax=epsilons.max())
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax1, ax2], pad=0.12, fraction=0.025)
    cbar.set_label(r"$\varepsilon$", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    # ── Combined legend ───────────────────────────────────────────────────────
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    # Add marker style legend entries
    extra_handles = [
        Line2D([0], [0], marker="o", color="gray", ls="none",
               markersize=7, label=r"$\mu$"),
        Line2D([0], [0], marker="D", color="gray", ls="none",
               markersize=7, label=r"$\sigma^2$"),
    ]
    ax1.legend(
        lines1 + lines2 + extra_handles,
        labels1 + labels2 + [r"$\mu$", r"$\sigma^2$"],
        loc="upper left", fontsize=9,
        frameon=True, framealpha=0.9, edgecolor="#cccccc"
    )

    # Remove right spine on ax1 to avoid overlap
    ax1.spines["right"].set_visible(False)

    fig.tight_layout()
    path = os.path.join(out_dir, "fig4_bias_acceptance_tradeoff.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Chargement des données...")
    df_bias, df_summary, df_gibbs, df_abc = load_data(INPUT_DIR)

    epsilons    = sorted(df_bias["epsilon"].unique())
    eps_palette = make_epsilon_palette(epsilons)

    print(f"  Grille epsilon : {epsilons}")
    print(f"  Datasets       : {df_gibbs['dataset_id'].nunique()}")
    print(f"  Echantillons ABC (poolés) : {len(df_abc):,}\n")

    print("Génération des figures...")
    plot_bias_vs_epsilon(df_bias, eps_palette, OUTPUT_DIR)
    plot_posterior_shift(df_gibbs, df_abc, eps_palette, REF_DS, OUTPUT_DIR)
    plot_bias_per_dataset(df_summary, eps_palette, OUTPUT_DIR)
    plot_bias_acceptance_tradeoff(df_bias, eps_palette, OUTPUT_DIR)

    print(f"\nDone ✓  —  figures dans : {OUTPUT_DIR}/")