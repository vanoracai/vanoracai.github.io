import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import chirp
import matplotlib

matplotlib.rc('text', usetex = True)
plt.rcParams["font.family"] = "Times New Roman"

# ── Parameters ──────────────────────────────────────────────────────────────
fs = 8000          # 8 kHz telephony sampling rate
T  = 0.08          # 80 ms total duration shown
N  = int(T * fs)   # 640 samples

np.random.seed(42)
n = np.arange(N)
t = n / fs

# ── Synthesise three phonetic segments ──────────────────────────────────────
# Segment boundaries (in samples)
seg1_end = 160   # 0–160: voiced vowel /a/  (quasi-periodic)
seg2_end = 340   # 160–340: unvoiced fricative /s/ (noise-like)
seg3_end = 640   # 340–640: voiced vowel /o/  (quasi-periodic)

signal = np.zeros(N)

# --- Voiced /a/ : sum of harmonics of F0=120 Hz + slight AM envelope
F0 = 120
env1 = np.hanning(seg1_end) * 0.9
harmonics = [1, 2, 3, 4]
amps      = [1.0, 0.6, 0.35, 0.18]
for k, a in zip(harmonics, amps):
    signal[:seg1_end] += a * np.sin(2*np.pi*k*F0*t[:seg1_end])
signal[:seg1_end] *= env1

# --- Unvoiced /s/ : bandpass noise (2–4 kHz)
noise = np.random.randn(seg2_end - seg1_end) * 0.3
# simple crude bandpass via diff (accentuates HF)
from scipy.signal import butter, sosfilt
sos = butter(4, [1800, 3800], btype='band', fs=fs, output='sos')
noise_bp = sosfilt(sos, noise)
noise_bp /= np.max(np.abs(noise_bp)) * 1.4   # normalise
env2 = np.hanning(seg2_end - seg1_end) * 0.55
signal[seg1_end:seg2_end] = noise_bp * env2

# --- Voiced /o/ : F0=100 Hz, different harmonic mix
F0b = 100
slice3 = seg3_end - seg2_end
env3 = np.hanning(slice3) * 0.85
t3   = np.arange(slice3) / fs
harmonics3 = [1, 2, 3, 5]
amps3      = [1.0, 0.5, 0.25, 0.08]
for k, a in zip(harmonics3, amps3):
    signal[seg2_end:seg3_end] += a * np.sin(2*np.pi*k*F0b*t3)
signal[seg2_end:seg3_end] *= env3

# ── Downsample for stem plot (every 4th sample → 160 visible stems) ─────────
step = 4
n_ds = n[::step]
s_ds = signal[::step]

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 3.6))
fig.patch.set_facecolor('white')

# Colour stems by segment
colors = []
for idx in n_ds:
    if idx < seg1_end:
        colors.append('#2166ac')   # blue  — voiced
    elif idx < seg2_end:
        colors.append('#d73027')   # red   — unvoiced
    else:
        colors.append('#2166ac')   # blue  — voiced

markerline, stemlines, baseline = ax.stem(n_ds, s_ds,
                                          linefmt='none',
                                          markerfmt='none',
                                          basefmt='k-')
# Draw each stem manually with per-sample colour
baseline.set_linewidth(0.8)
ax.axhline(0, color='black', linewidth=0.7, zorder=0)
for xi, yi, c in zip(n_ds, s_ds, colors):
    ax.plot([xi, xi], [0, yi], color=c, linewidth=0.8, alpha=0.85)
    ax.plot(xi, yi, 'o', color=c, markersize=1.8, alpha=0.9)

# ── Segment annotations ───────────────────────────────────────────────────────
ax.axvspan(0,        seg1_end, alpha=0.07, color='#2166ac', zorder=0)
ax.axvspan(seg1_end, seg2_end, alpha=0.07, color='#d73027', zorder=0)
ax.axvspan(seg2_end, seg3_end, alpha=0.07, color='#2166ac', zorder=0)

ax.axvline(seg1_end, color='gray', linewidth=0.9, linestyle='--', alpha=0.7)
ax.axvline(seg2_end, color='gray', linewidth=0.9, linestyle='--', alpha=0.7)

# Bracket-style labels above the plot
ymax = 1.05
label_cfg = dict(ha='center', va='bottom', fontsize=9.5, fontfamily='DejaVu Sans')
ax.text((0+seg1_end)/2,        ymax, 'Voiced /a/\n(quasi-periodic)', color='#2166ac', **label_cfg)
ax.text((seg1_end+seg2_end)/2, ymax, 'Unvoiced /s/\n(noise-like)',   color='#d73027', **label_cfg)
ax.text((seg2_end+seg3_end)/2, ymax, 'Voiced /o/\n(quasi-periodic)', color='#2166ac', **label_cfg)

# ── Axes labels & ticks ───────────────────────────────────────────────────────
ax.set_xlabel(r'Sample index  $n$', fontsize=11)
ax.set_ylabel(r'Amplitude  $x(n)$', fontsize=11)
ax.set_title(r'Speech signal sampled at $f_s = 8\,\mathrm{kHz}$  '
             r'(discrete-time sequence)',
             fontsize=11.5, pad=28)

# Secondary x-axis in milliseconds
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
ms_ticks = np.arange(0, T*1000+1, 10)
ax2.set_xticks(ms_ticks * fs / 1000)
ax2.set_xticklabels([f'{m:.0f}' for m in ms_ticks], fontsize=8)
ax2.set_xlabel('Time  (ms)', fontsize=9, labelpad=3)

ax.set_xlim(-8, N+5)
ax.set_ylim(-1.25, 1.55)
ax.tick_params(labelsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Legend
patch_v = mpatches.Patch(color='#2166ac', alpha=0.6, label='Voiced (vowel)')
patch_u = mpatches.Patch(color='#d73027', alpha=0.6, label='Unvoiced (fricative)')
ax.legend(handles=[patch_v, patch_u], loc='lower right', fontsize=9, framealpha=0.8)

plt.tight_layout()
plt.savefig('../Fig/Chapter_1/fig_1_1.png',
            dpi=180, bbox_inches='tight', facecolor='white')
print("Saved.")
