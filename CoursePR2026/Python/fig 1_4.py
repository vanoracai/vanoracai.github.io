"""
Figure 1.4: Radar Signals
(left)  LFM chirp with linearly increasing instantaneous frequency
(right) Frequency-diversity radar with stepped carrier frequencies across pulses
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib

matplotlib.rc('text', usetex = True)
plt.rcParams["font.family"] = "Times New Roman"

plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.linewidth": 1.0,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))

# =====================================================================
# Left panel: LFM chirp — time-domain waveform + instantaneous frequency
#   x(t) = rect(t/T) * exp(j*pi*mu*t^2),  mu = B/T
# =====================================================================
T = 1.0                 # pulse duration
B = 20.0                # swept bandwidth (Hz)
mu = B / T               # chirp rate
fs = 2000
t = np.linspace(0, 1.25 * T, fs)
pulse_mask = t <= T

chirp_untruncated = np.sin(np.pi * mu * t ** 2)
chirp = np.where(pulse_mask, chirp_untruncated, 0.0)  # rectangularly truncated waveform
f_inst = np.where(pulse_mask, mu * t, np.nan)         # only defined inside the pulse

ax1.plot(t, chirp, color="#1f4e8c", linewidth=0.9)
ax1.axhline(0, color="gray", linewidth=0.5, zorder=0)
ax1.axvline(T, color="gray", linewidth=0.8, linestyle=":", zorder=0)
ax1.axvspan(T, t[-1], color="gray", alpha=0.08, zorder=-1)
ax1.set_xlim(0, t[-1])
ax1.set_ylim(-1.3, 1.3)
ax1.set_xlabel("Time $t$")
ax1.set_ylabel("Amplitude", color="#1f4e8c")
ax1.tick_params(axis="y", labelcolor="#1f4e8c")
ax1.set_title("LFM Chirp: Truncated Time Domain", fontsize=12)

ax1b = ax1.twinx()
ax1b.plot(t, f_inst, color="#c0392b", linewidth=1.8, linestyle="--")
ax1b.set_ylabel(r"Instantaneous frequency $f_i(t)=\mu t$", color="#c0392b")
ax1b.tick_params(axis="y", labelcolor="#c0392b")
ax1b.set_ylim(0, B * 1.1)

ax1.text(
    0.03, 0.95,
    r"$s(t)=\mathrm{rect}\!\left(\frac{t-T/2}{T}\right)\sin(\pi\mu t^2)$",
    transform=ax1.transAxes, fontsize=10, va="top", ha="left",
    bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.85),
)

ax1.text(
    T + 0.02 * T, -1.12,
    "truncated",
    fontsize=9, color="gray", va="bottom", ha="left",
)

# =====================================================================
# Right panel: frequency-diversity radar — stepped carrier frequency
#   across successive pulses
# =====================================================================
n_pulses = 6
pulse_width = 0.6
gap = 0.4
freqs = np.arange(1, n_pulses + 1) * 8   # f_1, f_2, ..., f_K (stepped carriers)
colors = plt.cm.viridis(np.linspace(0.15, 0.85, n_pulses))

for i, f in enumerate(freqs):
    t0 = i * (pulse_width + gap)

    # pulse envelope (rectangle) at its carrier frequency level
    rect = patches.Rectangle(
        (t0, f - 2.2), pulse_width, 4.4,
        facecolor=colors[i], edgecolor="black", linewidth=0.8, alpha=0.85, zorder=2,
    )
    ax2.add_patch(rect)

    # a few oscillation cycles drawn inside the rectangle to suggest the carrier
    tt = np.linspace(t0, t0 + pulse_width, 200)
    osc = 1.6 * np.sin(2 * np.pi * 3 * (tt - t0) / pulse_width)
    ax2.plot(tt, f + osc, color="black", linewidth=0.7, zorder=3)

# dotted staircase guide connecting successive carrier frequencies
centers = np.array([
    i * (pulse_width + gap) + pulse_width / 2
    for i in range(n_pulses)
])
ax2.step(centers, freqs, where="mid", color="#c0392b", linewidth=1.2,
          linestyle=":", zorder=1)

ax2.set_xlim(0, n_pulses * (pulse_width + gap))
ax2.set_ylim(0, freqs[-1] + 8)
ax2.set_xlabel("Pulse index $k$")
ax2.set_ylabel("Carrier frequency $f_k$")
ax2.set_title("Frequency-Diversity Radar: Stepped Carriers", fontsize=12)
ax2.set_xticks(centers)
ax2.set_xticklabels([f"Pulse {i + 1}" for i in range(n_pulses)], fontsize=8, rotation=30)
ax2.set_yticks(freqs)
ax2.set_yticklabels([f"$f_{{{i+1}}}$" for i in range(n_pulses)])

fig.suptitle(
    "Figure 1.4: Radar Signals \u2014 LFM Chirp vs. Frequency-Diversity Waveform",
    fontsize=13, y=1.03,
)
fig.tight_layout()
fig.savefig("../Fig/Chapter_1/fig_1_4.png", dpi=220, bbox_inches="tight")
