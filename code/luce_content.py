"""Content values for the Luce newsreel: token positions of the speech text (Opera Omnia)
placed approximately on the time axis. Between anchors, tokens are placed in proportion to
cumulative voiced time.

Usage: python3 luce_content.py BROWSER.json luce_text_events.json OUT_CAPTIONS.json
"""
import sys, json
import numpy as np

b = json.load(open(sys.argv[1])); tx = json.load(open(sys.argv[2]))
A = np.array(b["A"], float); A = A[np.argsort(A[:, 0])]; A = A[np.r_[True, np.diff(A[:, 0]) > 0]]
t, rms, f0 = A[:, 0], A[:, 1], A[:, 2]
voiced = (f0 > 0) & (rms > np.percentile(rms, 30))
cum = np.concatenate([[0], np.cumsum(voiced[1:] * np.diff(t))])  # cumulative voiced time
def t2c(x): return np.interp(x, t, cum)
def c2t(c): return np.interp(c, cum, t)
anc = tx["anchors"]
idx = np.array([a[0] for a in anc], float); ct = np.array([t2c(a[1]) for a in anc])
n = tx["n_tokens"]
rate = (ct[-1] - ct[-2]) / (idx[-1] - idx[-2])
idx = np.append(idx, n - 1); ct = np.append(ct, ct[-1] + rate * (n - 1 - idx[-2]))
def tok_time(i): return float(c2t(np.interp(i, idx, ct)))
CAT = {"w": "we", "y": "you", "t": "they", "i": "i"}
ev, dropped = [], 0
tend = float(t[voiced][-1]) if voiced.any() else float(t[-1])
for item in tx["events_idx"].split():
    i, c = int(item[:-1]), CAT[item[-1]]
    x = tok_time(i)
    if i > anc[-1][0] and x >= tend - 0.5:
        dropped += 1; continue
    ev.append([round(x, 2), c])
tt = [tok_time(i) for i in range(n) if not (i > anc[-1][0] and tok_time(i) >= tend - 0.5)]
spans = []
for x in tt:
    if spans and x - spans[-1][1] < 0.8: spans[-1][1] = x + 0.3
    else: spans.append([x, x + 0.3])
json.dump({"events": ev, "speech_spans": [[round(a, 2), round(c, 2)] for a, c in spans],
           "method": "word positions of the speech text (Opera Omnia) placed approximately on the time axis, in proportion to voiced time between five points that clearly match the automatic subtitles. %d markers after the last point could not be placed and are not shown." % dropped}, open(sys.argv[3], "w"))
print("dropped", dropped, "events", len(ev), "first", ev[:3], "last", ev[-2:])
