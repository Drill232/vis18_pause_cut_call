"""Sound layer, four stages.
The only input is a measurement file (data/m_*.json). Original sound is not read anywhere.

stage 1  level and tilt   noise shaped every frame by measured loudness and spectral tilt. f0 is not used.
stage 2  level            tilt fixed to one common value for all three materials (median).
stage 3  presence         the contour of loudness is erased: any sound, speech or not, at one level.
stage 4  speech and pause sound not produced by the speaker is erased as well: only speech and pause remain.
"""
import json, os, subprocess
import numpy as np
import soundfile as sf
import operators as op
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent

D = f"{ROOT}/data"
O = f"{ROOT}/out/sound_layer"
os.makedirs(O, exist_ok=True)

MATS = [("luce_B0539", "1934"), ("mattarella_2022", "2022"), ("meloni_20230129", "2023")]
RUNGS = {1: "level_tilt", 2: "level", 3: "presence", 4: "speech_pause"}
SILENT = -80.0


def rungs_for(m, common_tilt):
    env = op.spectral_envelope(m)                 # t, level_db, tilt
    speech = op.time_skeleton(m)["speech"] > 0
    crowd = op.outside_speech(m)["crowd"] > 0
    flat = np.full_like(env["tilt"], common_tilt)
    return {
        1: {"t": env["t"], "level_db": env["level_db"], "tilt": env["tilt"]},
        2: {"t": env["t"], "level_db": env["level_db"], "tilt": flat},
        3: {"t": env["t"], "level_db": np.where(speech | crowd, -6.0, SILENT), "tilt": flat},
        4: {"t": env["t"], "level_db": np.where(speech, -6.0, SILENT), "tilt": flat},
    }, {"speech": speech.mean(), "crowd": crowd.mean()}


def normalize(x, target_db=-20.0):
    """Sets the mean level of the sounding parts to one value (-20 dBFS), so that materials and stages can be compared."""
    fr = int(op.SR * 0.05)
    n = len(x) // fr
    e = np.sqrt((x[: n * fr].reshape(n, fr) ** 2).mean(1) + 1e-12)
    act = e[20 * np.log10(e) > 20 * np.log10(e.max()) - 40]
    g = 10 ** (target_db / 20) / (np.sqrt((act ** 2).mean()) + 1e-12)
    y = x * g
    pk = np.abs(y).max()
    return y * (0.95 / pk) if pk > 0.95 else y


def mp3(wav, out):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav, "-ac", "1",
                    "-b:a", "160k", out], check=True)


def main():
    ms = {k: op.load(f"{D}/m_{k}.json") for k, _ in MATS}
    common = float(np.median(np.concatenate([m["tilt"] for m in ms.values()])))
    audio, info = {}, {"common_tilt_db_per_khz": common, "sr": op.SR,
                       "hop_s": op.HOP, "materials": {}}
    for k, yr in MATS:
        rs, frac = rungs_for(ms[k], common)
        info["materials"][k] = {"duration_s": ms[k]["dur"], **{f + "_frac": round(float(v), 4) for f, v in frac.items()}}
        for r, env in rs.items():
            y = normalize(op.render_envelope(env, seed=r))
            y = y[: int(ms[k]["dur"] * op.SR)]
            audio[(k, r)] = y
            base = f"{O}/sound_layer_{yr}_stage{r}_{RUNGS[r]}"
            sf.write(base + ".wav", y, op.SR, subtype="PCM_16")
            mp3(base + ".wav", base + ".mp3")
            print("wrote", base)

    json.dump(info, open(f"{O}/sound_layer_parameters.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(info, ensure_ascii=False, indent=1)[:1500])


if __name__ == "__main__":
    main()
