"""
fig_3_0a.py
===========
Generate Figure 3.0a for Chapter 3 (Digital Filter Structures).

Contrasts the impulse responses of:
  Left  — 8-tap FIR  : h(n) = 0.8^n  for n = 0..7,  zero elsewhere
  Right — 1st-order IIR: h(n) = 0.8^n u(n),  decaying but theoretically infinite

Usage:
    python fig_3_0a.py            # saves fig_3_0a.png + fig_3_0a.pdf
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
import matplotlib

matplotlib.rc('text', usetex = True)
plt.rcParams["font.family"] = "Times New Roman"

# ── Style ──────────────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "text.usetex": False,           # set True if a LaTeX install is available
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── Parameters ─────────────────────────────────────────────────────────────────
a = 0.8          # pole / decay coefficient (both filters share the same α)
N_FIR = 8       # number of FIR taps (nonzero samples: 0 … N_FIR-1)
N_IIR = 24      # how many IIR samples to plot  (rest shown as "…")

n_fir = np.arange(N_FIR)
h_fir = a ** n_fir                  # decaying FIR coefficients  [0..7]

n_iir = np.arange(N_IIR)
h_iir = a ** n_iir                  # h(n) = 0.8^n u(n)

# Colour palette
C_FIR  = "#2C6FAC"   # steel blue — FIR stems
C_IIR  = "#C0392B"   # brick red  — IIR stems
C_ZERO = "#BDC3C7"   # light grey — zero-value stems (FIR tail)
C_MRK  = "white"     # marker face

# ── Canvas ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8),
                          gridspec_kw={"wspace": 0.38})

# ══════════════════════════════════════════════════════════════════════════════
# LEFT  — FIR
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[0]

# Nonzero stems
mc_fir, sl_fir, bl_fir = ax.stem(n_fir, h_fir, linefmt=C_FIR,
                                   markerfmt="o", basefmt=" ")
plt.setp(sl_fir, linewidth=1.6)
plt.setp(mc_fir, markersize=7, markerfacecolor=C_MRK,
         markeredgecolor=C_FIR, markeredgewidth=1.8)

# Zero region (n = 8 … N_IIR-1) — show a few samples to emphasise "zeroed out"
n_zero = np.arange(N_FIR, N_FIR + 6)
mc_z, sl_z, bl_z = ax.stem(n_zero, np.zeros(len(n_zero)),
                             linefmt=C_ZERO, markerfmt="o", basefmt=" ")
plt.setp(sl_z, linewidth=1.2, linestyle="--")
plt.setp(mc_z, markersize=6, markerfacecolor=C_ZERO,
         markeredgecolor=C_ZERO, markeredgewidth=1.2)

# Baseline
ax.axhline(0, color="black", linewidth=0.8, zorder=0)

# Annotations
ax.annotate("", xy=(N_FIR - 0.15, -0.055), xytext=(0.15, -0.055),
            arrowprops=dict(arrowstyle="<->", color="#555555", lw=1.2))
ax.text(3.8, -0.10, "$M = 8$ taps", ha="center", va="top",
        fontsize=10, color="#555555")

ax.text(N_FIR + 0.3, 0.13, r"$h(n)=0$ for $n\geq 8$",
        fontsize=9.5, color=C_ZERO, va="center")

# Complexity callout box
ax.text(0.97, 0.97,
        "Cost: 8 mult/sample",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=10, color=C_FIR,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#EAF3FB",
                  edgecolor=C_FIR, linewidth=1.1))

ax.set_title(r"FIR  —  finite impulse response", pad=7)
ax.set_xlabel(r"$n$  (sample index)")
ax.set_ylabel(r"$h(n)$")
ax.set_xlim(-0.8, N_FIR + 7)
ax.set_ylim(-0.18, 1.12)
ax.xaxis.set_major_locator(MultipleLocator(4))
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(0.25))

# ══════════════════════════════════════════════════════════════════════════════
# RIGHT — IIR
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[1]

mc_iir, sl_iir, bl_iir = ax.stem(n_iir, h_iir, linefmt=C_IIR,
                                   markerfmt="o", basefmt=" ")
plt.setp(sl_iir, linewidth=1.6)
plt.setp(mc_iir, markersize=7, markerfacecolor=C_MRK,
         markeredgecolor=C_IIR, markeredgewidth=1.8)

# Baseline
ax.axhline(0, color="black", linewidth=0.8, zorder=0)

# Equation label
ax.text(13.5, 0.78, r"$h(n) = (0.8)^n\, u(n)$",
        fontsize=10.5, color=C_IIR, ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FDECEA",
                  edgecolor=C_IIR, linewidth=1.0))

# "continues …" ellipsis
ax.text(N_IIR + 0.4, 0.012, r"$\cdots$", fontsize=14,
        va="center", color=C_IIR)

# Complexity callout box
ax.text(0.97, 0.97,
        "Cost: 2 mult/sample",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=10, color=C_IIR,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#FDECEA",
                  edgecolor=C_IIR, linewidth=1.1))

ax.set_title(r"IIR  —  infinite impulse response  ($a = 0.8$)", pad=7)
ax.set_xlabel(r"$n$  (sample index)")
ax.set_ylabel(r"$h(n)$")
ax.set_xlim(-0.8, N_IIR + 1.5)
ax.set_ylim(-0.06, 1.12)
ax.xaxis.set_major_locator(MultipleLocator(8))
ax.xaxis.set_minor_locator(MultipleLocator(2))
ax.yaxis.set_major_locator(MultipleLocator(0.25))

# ── Figure-level caption ───────────────────────────────────────────────────────
fig.text(
    0.5, -0.01,
    "Figure 3.0a   Impulse responses contrasted.  "
    "Both use the same decay coefficient α = 0.8.\n"
    "FIR (left): 8 nonzero taps, exactly zero for n ≥ 8.  "
    "IIR (right): h(n) = 0.8ⁿ u(n), theoretically nonzero for all n ≥ 0.",
    ha="center", va="top", fontsize=9.5, color="#444444",
    style="italic"
)

fig.savefig("../Fig/Chapter_1/fig_3_0a.png", dpi=220, bbox_inches="tight")
