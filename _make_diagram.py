import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

fig = plt.figure(figsize=(18, 14))

C_R  = "#C0392B"
C_M  = "#2471A3"
C_SH = "#BDC3C7"

def box(ax, x, y, w, h, text, color, fontsize=9.5, text_color="white"):
    bb = FancyBboxPatch((x - w/2, y - h/2), w, h,
                        boxstyle="round,pad=0.12", linewidth=1.4,
                        edgecolor="white", facecolor=color, zorder=3)
    ax.add_patch(bb)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, color=text_color, zorder=4,
            multialignment="center", fontweight="bold")

def diamond(ax, x, y, w, h, text, color, fontsize=9):
    pts = np.array([[x, y+h/2], [x+w/2, y], [x, y-h/2], [x-w/2, y]])
    poly = plt.Polygon(pts, closed=True, facecolor=color,
                       edgecolor="white", lw=1.4, zorder=3)
    ax.add_patch(poly)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, color="white", zorder=4,
            fontweight="bold", multialignment="center")

def arrow(ax, x1, y1, x2, y2, color="#555", lw=1.6, ls="-"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=14,
                                linestyle=ls))

# ─────────────────────────────────────────────────────────────────────────────
# LEFT:  Reject-ABC
# ─────────────────────────────────────────────────────────────────────────────
ax1 = fig.add_axes([0.02, 0.04, 0.44, 0.88])
ax1.set_xlim(0, 10); ax1.set_ylim(0, 22); ax1.axis("off")
ax1.set_facecolor("#FAFBFC")
ax1.set_title("Reject-ABC", fontsize=14, fontweight="bold", color=C_R, pad=8)

bw, bh = 7.0, 1.1

# y positions of each step
Y = dict(start=20.5, draw=18.8, sim=17.1, w1=15.4, dec1=13.7,
         keep=12.0, dec2=10.3, stats=8.6, out=6.9)

box(ax1,    5, Y["start"], bw, bh, "START\nObserved data y*,  tolerance ε",
    "#2C3E50", fontsize=9.5)
box(ax1,    5, Y["draw"],  bw, bh,
    "Draw batch of 4096 proposals\n(μ, log σ²)  ~  Prior  N(0,1)",
    C_SH, text_color="#2C3E50")
box(ax1,    5, Y["sim"],   bw, bh,
    "Simulate Y_sim for each proposal\n(JAX vmap — 4096 in parallel)",
    C_R)
box(ax1,    5, Y["w1"],    bw, bh,
    "Compute  W₁(y*, Y_sim)  for each\n(sort both, mean |gap|)",
    C_R)
diamond(ax1,5, Y["dec1"], 7.2, 1.3, "W₁  ≤  ε  ?", "#E67E22")
box(ax1,    5, Y["keep"],  bw, bh,
    "Keep accepted  (μ, σ²)\nadd to posterior sample list",
    C_R)
diamond(ax1,5, Y["dec2"], 7.2, 1.3,
        "n_accepted ≥ 200\nor proposals ≥ 500 000 ?", "#E67E22")
box(ax1,    5, Y["stats"], bw, bh,
    "Compute posterior means  E[μ],  E[σ²]",
    C_SH, text_color="#2C3E50")
box(ax1,    5, Y["out"],   bw, bh,
    "OUTPUT — 200 iid posterior samples", "#1E8449")

# main downward arrows
pairs = [("start","draw"),("draw","sim"),("sim","w1"),("w1","dec1"),
         ("dec1","keep"),("keep","dec2"),("dec2","stats"),("stats","out")]
diamonds_ = {"dec1","dec2"}
for a_,b_ in pairs:
    y1 = Y[a_] - (0.65 if a_ in diamonds_ else bh/2)
    y2 = Y[b_] + (0.65 if b_ in diamonds_ else bh/2)
    arrow(ax1, 5, y1, 5, y2)

# "No" from dec1: loop right side back to draw
ax1.plot([5+7.2/2, 8.6], [Y["dec1"], Y["dec1"]], color=C_R, lw=1.6)
ax1.plot([8.6,     8.6],  [Y["dec1"], Y["draw"]],  color=C_R, lw=1.6)
ax1.annotate("", xy=(5+bw/2, Y["draw"]), xytext=(8.6, Y["draw"]),
             arrowprops=dict(arrowstyle="-|>", color=C_R, lw=1.6, mutation_scale=14))
ax1.text(9.3, (Y["dec1"]+Y["draw"])/2, "No\n→ next\nbatch",
         color=C_R, fontsize=8, ha="center", va="center")
ax1.text(3.8, Y["dec1"]-0.9, "Yes ↓", color="#E67E22", fontsize=8.5, ha="center")

# "No" from dec2: right side back to draw (dashed)
ax1.plot([5+7.2/2, 9.2], [Y["dec2"], Y["dec2"]], color="#888", lw=1.4, ls="--")
ax1.plot([9.2,      9.2], [Y["dec2"], Y["draw"]],  color="#888", lw=1.4, ls="--")
ax1.annotate("", xy=(5+bw/2, Y["draw"]), xytext=(9.2, Y["draw"]),
             arrowprops=dict(arrowstyle="-|>", color="#888", lw=1.4, mutation_scale=14))
ax1.text(3.8, Y["dec2"]-0.9, "Yes ↓", color="#E67E22", fontsize=8.5, ha="center")

ax1.text(5, 5.4,
         "Cost  ∝  1/ε\n(tiny acceptance region → need many proposals)",
         ha="center", fontsize=9, color=C_R,
         bbox=dict(boxstyle="round,pad=0.4", fc="#FADBD8", ec=C_R, lw=1.2))

# ─────────────────────────────────────────────────────────────────────────────
# RIGHT: MCMC-ABC
# ─────────────────────────────────────────────────────────────────────────────
ax2 = fig.add_axes([0.54, 0.04, 0.44, 0.88])
ax2.set_xlim(0, 10); ax2.set_ylim(0, 22); ax2.axis("off")
ax2.set_facecolor("#FAFBFC")
ax2.set_title("MCMC-ABC  (Marjoram et al. 2003)", fontsize=14,
              fontweight="bold", color=C_M, pad=8)

Y2 = dict(start=20.5, init=18.8, prop=17.1, sim=15.4, dec=13.7,
          acc=12.0, rej=10.3, cnt=8.6, burn=7.0, stats=5.3, out=3.6)

box(ax2, 5, Y2["start"], bw, bh, "START\nObserved data y*,  tolerance ε",
    "#2C3E50", fontsize=9.5)
box(ax2, 5, Y2["init"],  bw, bh,
    "Find valid θ₀ from prior\nwith  W₁(y_sim(θ₀), y*) ≤ ε",
    C_SH, text_color="#2C3E50")
box(ax2, 5, Y2["prop"],  bw, bh,
    "Propose  θ' = θ + δ·Z,   Z ~ N(0,I)\n(random walk on  (μ, log σ))",
    C_M)
box(ax2, 5, Y2["sim"],   bw, bh,
    "Simulate Y_sim ~ model(θ')\nCompute  W₁(y*, Y_sim)",
    C_M)
diamond(ax2, 5, Y2["dec"], 7.2, 1.4,
        "W₁ ≤ ε  AND\nlog U < log π(θ')/π(θ) ?", "#E67E22")
box(ax2, 5, Y2["acc"],  bw, bh, "Accept:  θ ← θ'\nAppend θ to chain", C_M)
box(ax2, 5, Y2["rej"],  bw, bh, "Reject:  keep θ\nAppend θ to chain",
    C_SH, text_color="#2C3E50")
diamond(ax2, 5, Y2["cnt"], 7.2, 1.3, "i < N_total = 5000 ?", "#E67E22")
box(ax2, 5, Y2["burn"],  bw, bh,
    "Discard first 2000 steps (burn-in)\nKeep last 3000",
    C_M)
box(ax2, 5, Y2["stats"], bw, bh,
    "Compute posterior means  E[μ],  E[σ²]",
    C_SH, text_color="#2C3E50")
box(ax2, 5, Y2["out"],   bw, bh,
    "OUTPUT — 3000 correlated posterior samples", "#1E8449")

# straight downward arrows
for a_, b_ in [("start","init"),("init","prop"),("prop","sim"),("sim","dec")]:
    y1 = Y2[a_] - (0.65 if a_ in {"dec","cnt"} else bh/2)
    y2 = Y2[b_] + (0.65 if b_ in {"dec","cnt"} else bh/2)
    arrow(ax2, 5, y1, 5, y2)

# from dec: "Yes" → acc
arrow(ax2, 5, Y2["dec"]-0.7, 5, Y2["acc"]+bh/2, color=C_M)
ax2.text(3.8, Y2["dec"]-1.0, "Yes ↓", color="#E67E22", fontsize=8.5, ha="center")

# from dec: "No" → rej (side branch)
ax2.plot([5+7.2/2, 8.5], [Y2["dec"], Y2["dec"]], color="#888", lw=1.6)
ax2.plot([8.5, 8.5], [Y2["dec"], Y2["rej"]+bh/2], color="#888", lw=1.6)
ax2.annotate("", xy=(5+bw/2, Y2["rej"]), xytext=(8.5, Y2["rej"]),
             arrowprops=dict(arrowstyle="-|>", color="#888", lw=1.6, mutation_scale=14))
ax2.text(9.3, (Y2["dec"]+Y2["rej"])/2, "No\n→ reject",
         color="#888", fontsize=8, ha="center", va="center")

# acc and rej both go to cnt
arrow(ax2, 5, Y2["acc"]-bh/2, 5, Y2["cnt"]+0.65, color=C_M)
arrow(ax2, 5, Y2["rej"]-bh/2, 5, Y2["cnt"]+0.65, color="#888")

# cnt "Yes" loops back to prop
ax2.plot([5+7.2/2, 9.0], [Y2["cnt"], Y2["cnt"]], color=C_M, lw=1.6)
ax2.plot([9.0, 9.0], [Y2["cnt"], Y2["prop"]], color=C_M, lw=1.6)
ax2.annotate("", xy=(5+bw/2, Y2["prop"]), xytext=(9.0, Y2["prop"]),
             arrowprops=dict(arrowstyle="-|>", color=C_M, lw=1.6, mutation_scale=14))
ax2.text(9.6, (Y2["cnt"]+Y2["prop"])/2, "Yes\n→ next\nstep",
         color=C_M, fontsize=8, ha="center", va="center")

# cnt "No" → burn
arrow(ax2, 5, Y2["cnt"]-0.65, 5, Y2["burn"]+bh/2)
ax2.text(3.8, Y2["cnt"]-1.0, "No ↓", color="#E67E22", fontsize=8.5, ha="center")

# burn → stats → out
arrow(ax2, 5, Y2["burn"]-bh/2, 5, Y2["stats"]+bh/2)
arrow(ax2, 5, Y2["stats"]-bh/2, 5, Y2["out"]+bh/2)

ax2.text(5, 2.2,
         "Cost  ≈  constant in ε\n(chain walks inside the W₁ ≤ ε region)",
         ha="center", fontsize=9, color=C_M,
         bbox=dict(boxstyle="round,pad=0.4", fc="#D6EAF8", ec=C_M, lw=1.2))

# ─────────────────────────────────────────────────────────────────────────────
fig.text(0.5, 0.97, "Algorithmic Process Diagrams — ABC Methods",
         ha="center", va="top", fontsize=16, fontweight="bold", color="#2C3E50")
fig.text(0.5, 0.935,
         r"Both target $\pi_\varepsilon(\theta\mid y^\star)\propto"
         r"\mathbf{1}[W_1(y^\star,y_\mathrm{sim})\leq\varepsilon]\,\pi(\theta)$",
         ha="center", va="top", fontsize=11, color="#555")

plt.savefig("benchmark_results/process_diagrams.png", dpi=150,
            bbox_inches="tight", facecolor="#F8F9FA")
print("saved → benchmark_results/process_diagrams.png")
