"""
Figure 3.0b — Frequency Response Anatomy
FIR: Hamming-windowed sinc, M=48, wc=0.4pi
IIR: 4th-order Butterworth, wc=0.4pi
Panels: Magnitude (dB) | Phase (raw numerical) | Group Delay
Phase shown as-is from np.unwrap — stopband jumps are numerical artefacts
(|H|≈0 => atan2 noise => unwrap adds ±2pi erroneously), useful for teaching.
"""

import numpy as np
from scipy.signal import freqz, firwin, butter, group_delay
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rc('text', usetex = True)
plt.rcParams["font.family"] = "Times New Roman"

M, wc = 48, 0.4
b_fir = firwin(M + 1, wc, window='hamming')
b_iir, a_iir = butter(4, wc, btype='low', analog=False)

N = 4096
w, H_fir = freqz(b_fir, [1.0], worN=N)
_,  H_iir = freqz(b_iir, a_iir, worN=N)
freq = w / np.pi

mag_fir = 20 * np.log10(np.maximum(np.abs(H_fir), 1e-12))
mag_iir = 20 * np.log10(np.maximum(np.abs(H_iir), 1e-12))

ph_fir = np.unwrap(np.angle(H_fir))
ph_iir = np.unwrap(np.angle(H_iir))

mask = freq < 0.6
slope = np.dot(freq[mask], ph_iir[mask]) / np.dot(freq[mask], freq[mask])
ph_iir_lin = slope * freq

w_gd, gd_fir = group_delay((b_fir, [1.0]), w=N)
_,    gd_iir  = group_delay((b_iir, a_iir), w=N)

plt.rcParams.update({'font.family': 'DejaVu Sans', 'mathtext.fontset': 'dejavusans'})
C_FIR = '#185FA5'
C_IIR = '#993C1D'
C_REF = '#888888'
A = 0.08

fig, axes = plt.subplots(3, 1, figsize=(10.5, 11), facecolor='white',
    gridspec_kw=dict(left=0.11, right=0.89, top=0.95, bottom=0.07, hspace=0.55))

def shade(ax):
    ax.axvspan(0,   0.4, color=C_FIR,    alpha=A, lw=0)
    ax.axvspan(0.4, 0.5, color='#C8A000', alpha=A, lw=0)
    ax.axvspan(0.5, 1.0, color=C_IIR,    alpha=A, lw=0)
    ax.axvline(0.4, color='#aaa', lw=0.8, ls='--')

def base(ax, ylabel, ylim, yticks, ytl=None):
    ax.set_xlim(0, 1)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_yticks(yticks)
    if ytl:
        ax.set_yticklabels(ytl, fontsize=9)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_xticklabels([f'{v:.1f}' if v else '0' for v in np.arange(0, 1.1, 0.1)], fontsize=9)
    ax.tick_params(labelsize=9)
    ax.grid(True, lw=0.35, color='#ccc')
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
        sp.set_color('#aaa')

# ── Panel 1: Magnitude ─────────────────────────────────────────────────────────
ax = axes[0]
shade(ax)
ax.axhline(-3, color='#aaa', lw=0.6, ls='--')
ax.plot(freq, mag_iir, color=C_IIR, lw=1.9, label='IIR (Butterworth N=4)')
ax.plot(freq, mag_fir, color=C_FIR, lw=1.9, label='FIR (Hamming M=48)')
base(ax, r'$|H(e^{j\omega})|$ (dB)', (-85, 8), [-80, -60, -40, -20, 0])
ax.text(0.20, -78, 'Passband', ha='center', fontsize=8.5, color=C_FIR)
ax.text(0.45, -78, 'Trans.',   ha='center', fontsize=8,   color='#997700')
ax.text(0.75, -78, 'Stopband', ha='center', fontsize=8.5, color=C_IIR)
ax.legend(loc='upper right', fontsize=8.5, framealpha=0.9, edgecolor='#ccc')

# ── Panel 2: Phase (raw numerical) ────────────────────────────────────────────
ax = axes[1]
ax2r = ax.twinx()
shade(ax)

ax.plot(freq, ph_fir / np.pi, color=C_FIR, lw=1.4, label='FIR (np.unwrap, raw)')
ax2r.plot(freq, ph_iir_lin / np.pi, color=C_REF, lw=1.0, ls='--', label='IIR linear ref.')
ax2r.plot(freq, ph_iir / np.pi,     color=C_IIR, lw=1.9,           label='IIR (np.unwrap, raw)')

fir_ymin = int(np.floor(np.nanmin(ph_fir / np.pi) / 2)) * 2
fir_ymax = 2
ax.set_ylim(fir_ymin, fir_ymax)
ax.set_yticks(range(fir_ymin, fir_ymax + 1, 2))
ax.set_yticklabels([f'{v}$\\pi$' if v else '0' for v in range(fir_ymin, fir_ymax + 1, 2)], fontsize=9)
ax.set_ylabel(r'FIR $\angle H(e^{j\omega})$ (rad)', fontsize=10, color=C_FIR)
ax.tick_params(axis='y', colors=C_FIR, labelsize=9)

ax2r.set_ylim(-2.7, 0.5)
ax2r.set_yticks(np.arange(-2.5, 0.1, 0.5))
ax2r.set_yticklabels([f'{v:.1f}$\\pi$' for v in np.arange(-2.5, 0.1, 0.5)], fontsize=9)
ax2r.set_ylabel(r'IIR $\angle H(e^{j\omega})$ (rad)', fontsize=10, color=C_IIR)
ax2r.tick_params(axis='y', colors=C_IIR, labelsize=9)
for sp in ax2r.spines.values():
    sp.set_linewidth(0.5)
    sp.set_color('#aaa')

ax.set_xlim(0, 1)
ax.set_xticks(np.arange(0, 1.1, 0.1))
ax.set_xticklabels([f'{v:.1f}' if v else '0' for v in np.arange(0, 1.1, 0.1)], fontsize=9)
ax.grid(True, lw=0.35, color='#ccc')
ax.tick_params(labelsize=9)
for sp in ax.spines.values():
    sp.set_linewidth(0.5)
    sp.set_color('#aaa')

jump_idx = np.where(
    (freq > 0.5) & (np.abs(np.diff(np.concatenate([[0], ph_fir / np.pi]))) > 1.5)
)[0]
if len(jump_idx):
    jx = freq[jump_idx[0]]
    ax.annotate('unwrap error\n($\\pm2\\pi$ jump)',
                xy=(jx, ph_fir[jump_idx[0]] / np.pi),
                xytext=(jx + 0.08, ph_fir[jump_idx[0]] / np.pi + 2),
                fontsize=8, color='#555',
                arrowprops=dict(arrowstyle='->', color='#999', lw=0.8))

note = ('Raw numerical output of np.unwrap(np.angle(H))\n'
        r'Stopband jumps: $|H|\approx0 \Rightarrow$ angle undefined'
        r' $\Rightarrow$ unwrap adds $\pm2\pi$ erroneously')
ax.text(0.015, 0.97, note, transform=ax.transAxes, fontsize=8, color='#555', va='top',
        bbox=dict(boxstyle='round,pad=0.4', fc='#fffef0', ec='#cccc88', lw=0.6))

lines_l, labs_l = ax.get_legend_handles_labels()
lines_r, labs_r = ax2r.get_legend_handles_labels()
ax.legend(lines_l + lines_r, labs_l + labs_r, loc='lower left',
          fontsize=8.5, framealpha=0.9, edgecolor='#ccc')

# ── Panel 3: Group Delay ───────────────────────────────────────────────────────
ax = axes[2]
shade(ax)
ax.axhline(M / 2, color=C_FIR, lw=0.9, ls='--', alpha=0.5)
ax.plot(freq, np.clip(gd_iir, -2, 36), color=C_IIR, lw=1.9, label='IIR (Butterworth N=4)')
ax.plot(freq, gd_fir,                  color=C_FIR, lw=1.9, label='FIR (Hamming M=48)')
base(ax, r'$\tau(\omega)$ (samples)', (0, 34), range(0, 35, 4))
ax.set_xlabel(r'$\omega/\pi$ (normalized frequency)', fontsize=11)
ax.text(0.55, M / 2 + 0.7, f'$M/2 = {M // 2}$ samples', fontsize=8.5, color=C_FIR)
ax.legend(loc='upper right', fontsize=8.5, framealpha=0.9, edgecolor='#ccc',
          bbox_to_anchor=(0.61, 0.99))

# ── Save ───────────────────────────────────────────────────────────────────────
fig.savefig("../Fig/Chapter_1/fig_3_0b.png", dpi=320, bbox_inches="tight")
