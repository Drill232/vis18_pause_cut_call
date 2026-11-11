"""Generative algorithm, fixed version.
Input: measurement files (data/m_*.json) and stage 1 of the sound layer. Original sound and image are not used.

time   the output lasts L, the duration of the 1934 material. Longer materials are cut into segments of
       length L, and all segments start at 0: at output time t, material times t, t+L, t+2L ... sound together.
image  vertical 9:16. Face area becomes a square, as a share of a 16:9 window fitted to the frame width.
       All squares share one centre; where they overlap, black and white are inverted (XOR).
       A cut in any layer inverts the whole frame for 0.2 s.
       Person markers appear for 0.4 s as horizontal lines at a fixed height per category. No words remain.
sound  stage 1 of the sound layer of each segment, summed with the same segmentation.
There is no random element: the same settings give the same output.
"""
import json, os, subprocess
import numpy as np
import soundfile as sf
import operators as op
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent

D = f"{ROOT}/data"
SND = f"{ROOT}/out/sound_layer"
O = f"{ROOT}/out/generative"
os.makedirs(O, exist_ok=True)

P = {   # settings. Changing them gives a different version.
    "W": 720, "H": 1280, "FPS": 25,
    "length_source": "luce_B0539",
    "cut_hold_s": 0.2,
    "mark_hold_s": 0.4,
    "mark_px": 3,
    "mark_y": {"i": 0.18, "we": 0.30, "you": 0.70, "they": 0.82},
    "layer_gain_db": {"luce_B0539": 0.0, "mattarella_2022": -3.0, "meloni_20230129": -3.0},
    "sound_rung": 1,
}
MATS = [("luce_B0539", "1934"), ("mattarella_2022", "2022"), ("meloni_20230129", "2023")]


def layers():
    L = op.load(f"{D}/m_{P['length_source']}.json")["dur"]
    out = []
    for k, yr in MATS:
        m = op.load(f"{D}/m_{k}.json")
        for f in range(int(np.ceil(m["dur"] / L))):
            out.append({"key": k, "yr": yr, "segment": f, "off": f * L, "m": m})
    return L, out


def face_at(m, tau):
    s = m["screen"]
    if tau >= m["dur"]:
        return 0.0
    i = max(0, np.searchsorted(s[:, 0], tau, "right") - 1)
    return s[i, 1]


def render_video(L, lay, silent_path):
    W, H, FPS = P["W"], P["H"], P["FPS"]
    win_area = W * (W * 9 / 16)
    cy = H // 2
    n = int(L * FPS)
    cuts = np.concatenate([l["m"]["cuts"][(l["m"]["cuts"] >= l["off"]) & (l["m"]["cuts"] < l["off"] + L)] - l["off"]
                           for l in lay])
    marks = []
    for l in lay:
        for t, c in l["m"]["events"]:
            if l["off"] <= t < l["off"] + L:
                marks.append((t - l["off"], c))
    marks = np.array([[t, list(P["mark_y"]).index(c)] for t, c in marks], float)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "gray",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "18", "-pix_fmt", "yuv420p", silent_path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    ys = [int(P["mark_y"][c] * H) for c in P["mark_y"]]
    for i in range(n):
        t = i / FPS
        ink = np.zeros((H, W), bool)
        for l in lay:
            a = face_at(l["m"], t + l["off"])
            if a > 0:
                s = int(round(np.sqrt(a * win_area)))
                y0, x0 = cy - s // 2, W // 2 - s // 2
                ink[y0:y0 + s, x0:x0 + s] ^= True
        if len(marks):
            on = marks[(marks[:, 0] <= t) & (t < marks[:, 0] + P["mark_hold_s"])]
            for _, c in on:
                y = ys[int(c)]
                ink[y:y + P["mark_px"], :] ^= True
        if np.any((cuts <= t) & (t < cuts + P["cut_hold_s"])):
            ink = ~ink
        p.stdin.write(np.where(ink, 0, 255).astype(np.uint8).tobytes())
    p.stdin.close()
    p.wait()
    return len(cuts), len(marks)


def render_sound(L, lay, path):
    SR = op.SR
    N = int(L * SR)
    mix = np.zeros(N)
    names = {1: "level_tilt", 2: "level", 3: "presence", 4: "speech_pause"}
    r = P["sound_rung"]
    cache = {}
    fade = int(0.05 * SR)
    for l in lay:
        if l["key"] not in cache:
            cache[l["key"]], _ = sf.read(f"{SND}/sound_layer_{l['yr']}_stage{r}_{names[r]}.wav")
        x = cache[l["key"]]
        a = int(l["off"] * SR)
        seg = x[a:a + N].copy()
        if len(seg) < N:
            seg[-fade:] *= np.linspace(1, 0, fade)
        mix[:len(seg)] += seg * 10 ** (P["layer_gain_db"][l["key"]] / 20)
    mix *= 0.89 / (np.abs(mix).max() + 1e-9)
    sf.write(path, mix, SR, subtype="PCM_16")


if __name__ == "__main__":
    L, lay = layers()
    silent = f"{O}/_video.mp4"
    wav = f"{O}/generative_fixed_version.wav"
    nc, nm = render_video(L, lay, silent)
    render_sound(L, lay, wav)
    final = f"{O}/generative_fixed_version.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", silent, "-i", wav, "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", final], check=True)
    os.remove(silent)
    info = dict(P, length_s=L, layers=[[l["yr"], l["segment"], round(l["off"], 2)] for l in lay],
                cuts_in_piece=int(nc), address_marks_in_piece=int(nm))
    json.dump(info, open(f"{O}/generative_parameters.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(info, ensure_ascii=False))
