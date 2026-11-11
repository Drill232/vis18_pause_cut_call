"""Layout figures. Black and white.

Time is laid out in lines of 60 seconds, read like a score. All materials use the same
length per second, so the number of lines is the difference in duration. Each line has
five lanes: content, loudness, pitch, tilt, image. Measured values are drawn frame by
frame, without summarising or smoothing.

Usage: python3 plot.py MEASURE.json OUT_BASENAME "Title" "Subtitle" ["Content method"]
"""
import sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

for f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
    try:
        fm.fontManager.addfont(f)
    except Exception:
        pass
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["svg.fonttype"] = "none"

LINE_S = 60.0
LANES = ["content", "loudness", "pitch", "tilt", "image"]
LANE_H = [0.8, 1.0, 0.8, 0.6, 0.9]
GLYPH = {"we": ("o", "black", "we"), "you": ("o", "white", "you, address to the audience"), "they": ("x", "black", "they"), "i": ("^", "black", "I")}

def main(path, out, title, subtitle, method=None):
    d = json.load(open(path))
    v, s, c = d["voice"], d["screen"], d["content"]
    t = np.array(v["t"])
    rms = np.array(v["rms_db"])
    f0 = np.array([np.nan if x is None else x for x in v["f0_hz"]], dtype=float)
    tilt = np.array(v["tilt_db_per_khz"])
    dur = max(s["duration_s"], t[-1] if len(t) else 0)
    nlines = int(np.ceil(dur / LINE_S))
    # normalisation range within each material (2nd to 98th percentile)
    r_lo, r_hi = np.percentile(rms, [2, 98])
    tl_lo, tl_hi = np.percentile(tilt, [2, 98])
    smp = np.array(s["samples"]) if s["samples"] else np.zeros((0, 3))
    cap_thr = np.percentile(smp[:, 2], 85) if len(smp) else 1
    face_hi = max(smp[:, 1].max() if len(smp) else 0, 1e-3)

    lane_total = sum(LANE_H) + 0.6
    fig_h = 2.0 + nlines * lane_total * 0.32
    fig = plt.figure(figsize=(16, fig_h))
    ax = fig.add_axes([0.07, 0.35 / fig_h, 0.91, 1 - 1.95 / fig_h])
    ax.set_xlim(0, LINE_S); ax.set_ylim(-nlines * lane_total, 0.2)
    ax.axis("off")
    fig.text(0.07, 1 - 0.45 / fig_h, title, fontsize=18, weight="bold")
    fig.text(0.07, 1 - 0.8 / fig_h, subtitle, fontsize=10)
    method = method or c.get("method")
    if method:
        fig.text(0.07, 1 - 1.15 / fig_h, "Times of the content lane: " + method, fontsize=8)

    for li in range(nlines):
        t0 = li * LINE_S
        top = -li * lane_total
        y = top
        bands = {}
        for name, hgt in zip(LANES, LANE_H):
            bands[name] = (y - hgt, y)
            ax.text(-0.6, y - hgt / 2, name, ha="right", va="center", fontsize=7)
            y -= hgt
        ax.text(-0.6, top + 0.12, f"{int(t0 // 60)}:{int(t0 % 60):02d}", ha="right", va="bottom", fontsize=7, weight="bold")
        ax.plot([0, LINE_S], [y, y], color="black", lw=0.4)
        m = (t >= t0) & (t < t0 + LINE_S)
        x = t[m] - t0
        # content: thin line for speech spans, glyphs for person markers
        b0, b1 = bands["content"]; mid = (b0 + b1) / 2
        for a_, b_ in c.get("speech_spans", []):
            if b_ >= t0 and a_ < t0 + LINE_S:
                ax.plot([max(a_, t0) - t0, min(b_, t0 + LINE_S) - t0], [mid, mid], color="black", lw=0.6)
        for tt, cat in c.get("events", []):
            if t0 <= tt < t0 + LINE_S:
                mk, fc, _ = GLYPH[cat]
                ax.scatter([tt - t0], [mid], marker=mk, s=22, facecolors=fc if mk != "x" else "black", edgecolors="black" if mk != "x" else None, linewidths=0.8, zorder=3)
        # loudness: envelope filled from below
        b0, b1 = bands["loudness"]
        hgt = np.clip((rms[m] - r_lo) / (r_hi - r_lo + 1e-9), 0, 1) * (b1 - b0) * 0.92
        ax.fill_between(x, b0, b0 + hgt, color="black", lw=0)
        # pitch: f0 dots (log 60~400 Hz)
        b0, b1 = bands["pitch"]
        fm_ = f0[m]
        ok = ~np.isnan(fm_)
        yy = b0 + (np.log(np.clip(fm_[ok], 60, 400)) - np.log(60)) / (np.log(400) - np.log(60)) * (b1 - b0)
        ax.scatter(x[ok], yy, s=0.6, color="black", lw=0)
        # tilt: thin line
        b0, b1 = bands["tilt"]
        yy = b0 + np.clip((tilt[m] - tl_lo) / (tl_hi - tl_lo + 1e-9), 0, 1) * (b1 - b0)
        ax.plot(x, yy, color="black", lw=0.35)
        # image: cuts (vertical lines), largest face area (step line), estimated subtitles (dots)
        b0, b1 = bands["image"]
        for ct in s["cuts_s"]:
            if t0 <= ct < t0 + LINE_S:
                ax.plot([ct - t0, ct - t0], [b0, b1], color="black", lw=0.9)
        if len(smp):
            sm = (smp[:, 0] >= t0) & (smp[:, 0] < t0 + LINE_S)
            sx = smp[sm, 0] - t0
            ax.step(sx, b0 + 0.15 + smp[sm, 1] / face_hi * (b1 - b0 - 0.3), where="post", color="black", lw=0.6)
            capm = smp[sm, 2] > cap_thr
            ax.scatter(sx[capm], np.full(capm.sum(), b0 + 0.06), marker="s", s=3, color="black", lw=0)
        # ten second ticks
        for k in range(0, int(LINE_S) + 1, 10):
            ax.plot([k, k], [y, y - 0.08], color="black", lw=0.4)

    # legend
    lg = "  ".join(f"{'●' if (g[0]=='o' and g[1]=='black') else '○' if g[0]=='o' else '×' if g[0]=='x' else '▲'} {g[2]}" for g in GLYPH.values())
    fig.text(0.07, 0.12 / fig_h, "content: " + lg + "   thin line: speech spans.  loudness: RMS, normalised to the 2nd~98th percentile of each material.  pitch: f0 (YIN estimate, octave errors possible), log 60~400 Hz.\n"
             "tilt: spectral tilt over 0.1~5 kHz.  image: vertical lines are cuts, step line is the largest face area, dots at the bottom are estimated subtitles.  One line = 60 s.",
             fontsize=7)
    fig.savefig(out + ".png", dpi=200); fig.savefig(out + ".svg")
    print("saved", out, nlines, "lines")

if __name__ == "__main__":
    main(*sys.argv[1:6])
