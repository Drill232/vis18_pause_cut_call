"""Operators on measured values.
The only input is a measurement file (data/m_*.json). Original sound and image are not used anywhere.
Every operator has the same form: it takes measured values, keeps what is specified, and erases the rest.
"""
import json
import numpy as np

SR = 22050       # resynthesis sample rate
HOP = 0.064      # measurement frame interval (s)


def load(path):
    d = json.load(open(path))
    v = d["voice"]
    t = np.array(v["t"], float)
    m = {
        "t": t,
        "rms": np.array(v["rms_db"], float),
        "f0": np.array([0 if x is None else x for x in v["f0_hz"]], float),
        "tilt": np.array(v["tilt_db_per_khz"], float),
        "events": d["content"].get("events", []),
        "screen": np.array(d["screen"]["samples"], float),   # t, face area, lower quarter edge density, motion
        "cuts": np.array(d["screen"]["cuts_s"], float),
        "dur": float(d["screen"]["duration_s"]),
    }
    return m


def _grid(m):
    return np.arange(0, m["dur"], HOP)


def _speech_mask(m):
    """Speech frames: frames within 0.5 s of a frame whose loudness is above the 35th percentile of the material and whose f0 was detected."""
    g = _grid(m)
    rms = np.interp(g, m["t"], m["rms"])
    f0 = np.interp(g, m["t"], m["f0"])
    voiced = (f0 > 0) & (rms > np.percentile(rms, 35))
    k = int(0.5 / HOP)
    return np.convolve(voiced.astype(float), np.ones(2 * k + 1), "same") > 0


# 1. time skeleton ---------------------------------------------------------------
def time_skeleton(m):
    """Keeps: durations of speech, pause and cuts. Erases: what is said, pitch and timbre of the voice, the image."""
    g = _grid(m)
    speech = _speech_mask(m)
    cut = np.zeros_like(g)
    cut[np.clip(np.searchsorted(g, m["cuts"]), 0, len(g) - 1)] = 1
    return {"t": g, "speech": speech.astype(float), "cut": cut}


# 2. spectral envelope -----------------------------------------------------------
def spectral_envelope(m):
    """Keeps: contour of loudness and spectral tilt. Erases: pitch, phonemes, all speech sound."""
    g = _grid(m)
    rms = np.interp(g, m["t"], m["rms"])
    tilt = np.interp(g, m["t"], m["tilt"])
    return {"t": g, "level_db": rms - np.percentile(rms, 98), "tilt": tilt}


def render_envelope(env, seed=0):
    """Shapes noise frame by frame to the measured tilt and loudness and overlaps the frames."""
    rng = np.random.default_rng(seed)
    n = int(SR * HOP) * 2
    hop = n // 2
    out = np.zeros(hop * (len(env["t"]) + 2))
    win = np.hanning(n)
    f = np.fft.rfftfreq(n, 1 / SR) / 1000.0
    for i, (lv, tl) in enumerate(zip(env["level_db"], env["tilt"])):
        spec = np.fft.rfft(rng.standard_normal(n) * win)
        shape = 10 ** ((tl * np.clip(f, 0.1, 8)) / 20.0)
        frame = np.fft.irfft(spec * shape / (np.sqrt((shape ** 2).mean()) + 1e-9), n) * win
        out[i * hop:i * hop + n] += frame * 10 ** (lv / 20.0)
    return out / (np.abs(out).max() + 1e-9) * 0.8


# 3. address times ---------------------------------------------------------------
PLACEHOLDER = {"we": 0, "you": 1, "they": 2, "i": 3}

def address_topology(m):
    """Keeps: when 'we', 'you', 'they', 'I' occur. Erases: which word it was and who was meant."""
    ev = np.array([[t, PLACEHOLDER[c]] for t, c in m["events"]], float).reshape(-1, 2)
    return {"t": ev[:, 0], "slot": ev[:, 1].astype(int)}


# 4. framing trajectory -----------------------------------------------------------
def framing_trajectory(m):
    """Keeps: share of the frame occupied by the face, cuts, amount of motion. Erases: the face, the place, lettering on screen."""
    s = m["screen"]
    return {"t": s[:, 0], "face": s[:, 1], "motion": s[:, 3], "cuts": m["cuts"]}


# 5. sound outside speech ---------------------------------------------------------
def outside_speech(m):
    """Keeps: spans that are loud without speech, and their level. Erases: what that sound was.
    Cheers, applause and music are not distinguished. Sound not produced by the speaker."""
    g = _grid(m)
    rms = np.interp(g, m["t"], m["rms"])
    speech = _speech_mask(m)
    loud = rms > np.percentile(rms, 60)
    crowd = (~speech) & loud
    level = np.where(crowd, rms - np.percentile(rms, 98), -80.0)
    return {"t": g, "crowd": crowd.astype(float), "level_db": level}


OPERATORS = [time_skeleton, spectral_envelope, address_topology, framing_trajectory, outside_speech]
