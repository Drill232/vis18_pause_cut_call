"""Converts the output of analyzer.js (JSON) into a measurement file (m_*.json).
Thumbnails are used only to measure face area and are not kept in the output.

Usage: python3 convert.py BROWSER.json CAPTIONS.json OUT.json
CAPTIONS.json: {"events": [[t, cat], ...], "speech_spans": [[a, b], ...], "method": "..."}
"""
import sys, json, base64
import numpy as np, cv2

b = json.load(open(sys.argv[1]))
cap = json.load(open(sys.argv[2])) if len(sys.argv) > 3 else {"events": [], "speech_spans": []}
out = sys.argv[-1]
A = np.array(b["A"], dtype=float)
A = A[np.argsort(A[:, 0])]
# drop spans where playback position went back (ads, seeking): keep monotonically increasing times only
keep = np.r_[True, np.diff(A[:, 0]) > 0]
A = A[keep]
voice = {"t": A[:, 0].round(3).tolist(), "rms_db": A[:, 1].tolist(),
         "f0_hz": [None if f <= 0 else f for f in A[:, 2]], "tilt_db_per_khz": A[:, 3].tolist()}
S = np.array(b["S"], dtype=float)
S = S[np.argsort(S[:, 0])]
cuts = [round(t, 2) for t, h in S[:, :2] if h > 0.35]
casc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
faces = {}
for t, data in b["T"]:
    img = cv2.imdecode(np.frombuffer(base64.b64decode(data), np.uint8), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    fs = casc.detectMultiScale(img, 1.1, 5, minSize=(30, 30))
    faces[round(t, 2)] = float(max((w * h for (_, _, w, h) in fs), default=0)) / (img.shape[0] * img.shape[1])
ft = np.array(sorted(faces))
rows = []
for t, h, mot, edge in S:
    fv = faces[ft[np.abs(ft - t).argmin()]] if len(ft) else 0.0
    rows.append([round(t, 2), round(fv, 4), round(edge, 4), round(mot, 2)])
screen = {"duration_s": round(float(b.get("duration") or A[-1, 0]), 2), "cuts_s": cuts, "samples": rows,
          "samples_cols": ["t", "largest_face_area_fraction", "lower_quarter_edge_density", "motion"]}
json.dump({"source": b.get("name"), "voice": voice, "screen": screen, "content": cap}, open(out, "w"))
print("ok", out, len(A), "audio frames,", len(cuts), "cuts,", len(cap.get("events", [])), "markers")
