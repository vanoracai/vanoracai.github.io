"""
Figure 1.3 — Baseband communication signals: constellation diagrams
QPSK / 8-PSK / 16-QAM / 64-QAM

Each subplot shows the symbol alphabet for one modulation scheme.
Gray-coded bit labels are shown for QPSK, 8-PSK, and 16-QAM (omitted
for 64-QAM to avoid clutter). Output is saved as fig_1_3.png to match
the figure-naming convention used elsewhere in the lecture notes.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rc('text', usetex = True)
plt.rcParams["font.family"] = "Times New Roman"


def gray_code(n_bits):
    """Return Gray-coded labels 0..2^n_bits-1 as binary strings."""
    n = 1 << n_bits
    return [format(i ^ (i >> 1), f"0{n_bits}b") for i in range(n)]


def psk_points(M):
    """Unit-energy M-PSK constellation points (start at 45 deg, like QPSK convention)."""
    k = np.arange(M)
    phase0 = np.pi / M  # rotate so symbols don't sit on the axes
    angles = 2 * np.pi * k / M + phase0
    return np.cos(angles) + 1j * np.sin(angles)


def qam_points(M):
    """Square M-QAM constellation (M must be a perfect square), unit average energy."""
    side = int(np.sqrt(M))
    levels = np.arange(-(side - 1), side, 2)  # e.g. side=4 -> [-3,-1,1,3]
    I, Q = np.meshgrid(levels, levels)
    pts = (I + 1j * Q).flatten()
    pts = pts / np.sqrt(np.mean(np.abs(pts) ** 2))  # normalize to unit avg energy
    return pts


def plot_constellation(ax, pts, title, bits_per_symbol, labels=None,
                        point_color="#1f6feb", label_fontsize=8):
    lim = np.max(np.abs(pts)) * 1.45

    # axes through origin
    ax.axhline(0, color="gray", linewidth=0.8, zorder=1)
    ax.axvline(0, color="gray", linewidth=0.8, zorder=1)

    # light reference circle for PSK schemes
    if np.allclose(np.abs(pts), np.abs(pts[0]), atol=1e-6):
        theta = np.linspace(0, 2 * np.pi, 200)
        r = np.abs(pts[0])
        ax.plot(r * np.cos(theta), r * np.sin(theta),
                color="lightgray", linewidth=0.8, zorder=1)

    ax.scatter(pts.real, pts.imag, s=70, color=point_color,
               edgecolor="black", linewidth=0.6, zorder=3)

    if labels is not None:
        for p, lab in zip(pts, labels):
            ax.annotate(lab, (p.real, p.imag),
                        textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=label_fontsize, zorder=4)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("In-phase (I)")
    ax.set_ylabel("Quadrature (Q)")
    ax.set_title(f"{title}  (M={len(pts)}, {bits_per_symbol} bits/symbol)",
                 fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4, zorder=0)


fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# --- QPSK (M=4) ---
M = 4
pts = psk_points(M)
plot_constellation(axes[0, 0], pts, "QPSK", int(np.log2(M)),
                    labels=gray_code(int(np.log2(M))), point_color="#1f6feb")

# --- 8-PSK (M=8) ---
M = 8
pts = psk_points(M)
plot_constellation(axes[0, 1], pts, "8-PSK", int(np.log2(M)),
                    labels=gray_code(int(np.log2(M))), point_color="#e8590c")

# --- 16-QAM (M=16) ---
M = 16
pts = qam_points(M)
plot_constellation(axes[1, 0], pts, "16-QAM", int(np.log2(M)),
                    labels=None, point_color="#2f9e44")

# --- 64-QAM (M=64) ---
M = 64
pts = qam_points(M)
plot_constellation(axes[1, 1], pts, "64-QAM", int(np.log2(M)),
                    labels=None, point_color="#9c36b5")

fig.suptitle("Baseband Communication Signals — Constellation Diagrams",
             fontsize=13, fontweight="bold", y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.97])

fig.savefig("../Fig/Chapter_1/fig_1_3.png", dpi=200, bbox_inches="tight")
