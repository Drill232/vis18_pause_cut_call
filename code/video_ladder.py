"""Video layer, four stages.
The only input is the image samples and cut times of a measurement file (data/m_*.json).
Original images are not used anywhere.

Drawing rules (same scale for all materials), black on white
  face      black square centred at (W/2, 0.40 H). area = share of the original frame occupied
            by the largest face x canvas area.
  band      black band at the bottom. thickness = edge density of the lower quarter x H/3.
  motion    displacement of square and band. amplitude = motion value x 0.25 px, directions from a fixed seed.
  cut       the whole frame is inverted for 0.2 s.

stage 1  face, band, motion, cut
stage 2  motion erased
stage 3  band erased
stage 4  face erased: only cuts remain

Each video carries the sound layer of the same stage. Aspect ratio is 16:9 for all materials.
"""
import json, os, subprocess, sys
import numpy as np
import operators as op
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent

D = f"{ROOT}/data"
SND = f"{ROOT}/out/sound_layer"
O = f"{ROOT}/out/video_layer"
os.makedirs(O, exist_ok=True)

MATS = [("luce_B0539", "1934"), ("mattarella_2022", "2022"), ("meloni_20230129", "2023")]
RUNGS = {1: "face_band_motion_cut", 2: "face_band_cut", 3: "face_cut", 4: "cut"}
SND_NAME = {1: "level_tilt", 2: "level", 3: "presence", 4: "speech_pause"}
W, H, FPS = 640, 360, 10
MOTION_PX = 0.25
CUT_HOLD = 0.2


class Source:
    def __init__(self, key):
        m = op.load(f"{D}/m_{key}.json")
        s = m["screen"]
        self.t, self.face, self.edge, self.motion = s[:, 0], s[:, 1], s[:, 2], s[:, 3]
        self.cuts = m["cuts"]
        self.dur = m["dur"]
        rng = np.random.default_rng(6)
        self.dirs = rng.uniform(0, 2 * np.pi, int(self.dur * FPS) + 10)

    def at(self, t):
        i = max(0, np.searchsorted(self.t, t, "right") - 1)
        return self.face[i], self.edge[i], self.motion[i]

    def in_cut(self, t):
        return np.any((self.cuts <= t) & (t < self.cuts + CUT_HOLD))


def frame(src, rung, i):
    t = i / FPS
    face, edge, motion = src.at(t)
    img = np.full((H, W), 255, np.uint8)
    dx = dy = 0
    if rung == 1:
        a = src.dirs[i]
        r = motion * MOTION_PX
        dx, dy = int(round(r * np.cos(a))), int(round(r * np.sin(a)))
    if rung <= 3 and face > 0:
        side = int(round(np.sqrt(face * W * H)))
        cx, cy = W // 2 + dx, int(H * 0.40) + dy
        y0, x0 = max(0, cy - side // 2), max(0, cx - side // 2)
        img[y0:min(H, cy - side // 2 + side), x0:min(W, cx - side // 2 + side)] = 0
    if rung <= 2:
        th = int(round(edge * H / 3))
        if th > 0:
            yb = H - int(H * 0.06) + dy
            img[max(0, yb - th):max(0, min(H, yb)), :] = 0
    if src.in_cut(t):
        img = 255 - img
    return img


def render(key, yr, rung):
    src = Source(key)
    n = int(src.dur * FPS)
    base = f"{O}/video_layer_{yr}_stage{rung}_{RUNGS[rung]}"
    wav = f"{SND}/sound_layer_{yr}_stage{rung}_{SND_NAME[rung]}.wav"
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-i", wav, "-map", "0:v", "-map", "1:a",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "128k", "-shortest", base + ".mp4"]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(n):
        p.stdin.write(frame(src, rung, i).tobytes())
    p.stdin.close()
    p.wait()
    return base + ".mp4"


if __name__ == "__main__":
    for key, yr in MATS:
        for r in (1, 2, 3, 4):
            print(render(key, yr, r), flush=True)
    json.dump({"w": W, "h": H, "fps": FPS, "motion_px_per_unit": MOTION_PX, "cut_hold_s": CUT_HOLD,
               "face": "square area = face_frac * W * H, centre (W/2, 0.40 H)",
               "band": "thickness = edge * H/3, bottom edge at 0.94 H",
               "sound": "each video carries the sound layer of the same stage"},
              open(f"{O}/video_layer_parameters.json", "w"), indent=1)
