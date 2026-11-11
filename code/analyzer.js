// In browser measurement. Plays the video muted on a YouTube watch page and measures
// sound (RMS, YIN f0, spectral tilt) and image (change of brightness distribution, motion,
// edge density of the lower quarter). Original sound and image are not stored. For face detection
// only, a 160x90 JPEG thumbnail is kept once per second; these thumbnails are not published.
(async () => {
  const NAME = window.__VIS_NAME || 'material';
  const v = document.querySelector('video');
  const player = document.querySelector('#movie_player') || document.querySelector('.html5-video-player');
  let started = false;
  v.muted = true; v.pause(); v.currentTime = 0;
  const ctx = new AudioContext({ sampleRate: 16000 });
  const src = ctx.createMediaStreamSource(v.captureStream());
  const N = 1024, SR = 16000;
  const proc = ctx.createScriptProcessor(N, 1, 1);
  const sink = ctx.createGain(); sink.gain.value = 0;
  src.connect(proc); proc.connect(sink); sink.connect(ctx.destination);
  // FFT setup
  const re = new Float64Array(N), im = new Float64Array(N);
  const win = new Float64Array(N).map((_, i) => 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (N - 1)));
  const lo = Math.ceil(100 * N / SR), hi = Math.floor(5000 * N / SR);
  const fk = []; for (let k = lo; k <= hi; k++) fk.push(k * SR / N / 1000);
  const fmean = fk.reduce((a, b) => a + b, 0) / fk.length;
  const fvar = fk.reduce((a, b) => a + (b - fmean) ** 2, 0);
  function fft() {
    for (let i = 1, j = 0; i < N; i++) { let bit = N >> 1; for (; j & bit; bit >>= 1) j ^= bit; j ^= bit; if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; } }
    for (let len = 2; len <= N; len <<= 1) { const a = -2 * Math.PI / len; for (let i = 0; i < N; i += len) for (let k = 0; k < len / 2; k++) {
      const c = Math.cos(a * k), s = Math.sin(a * k); const xr = re[i + k + len / 2] * c - im[i + k + len / 2] * s, xi = re[i + k + len / 2] * s + im[i + k + len / 2] * c;
      re[i + k + len / 2] = re[i + k] - xr; im[i + k + len / 2] = im[i + k] - xi; re[i + k] += xr; im[i + k] += xi; } }
  }
  // YIN (60~400 Hz)
  const tmin = Math.floor(SR / 400), tmax = Math.ceil(SR / 60), d = new Float64Array(tmax + 1);
  function yin(x) {
    const W = N - tmax; d[0] = 1; let run = 0;
    for (let tau = 1; tau <= tmax; tau++) { let s = 0; for (let i = 0; i < W; i++) { const df = x[i] - x[i + tau]; s += df * df; } run += s; d[tau] = run ? s * tau / run : 1; }
    for (let tau = tmin; tau < tmax; tau++) if (d[tau] < 0.15 && d[tau] <= d[tau + 1]) {
      const a = d[tau - 1], b = d[tau], c = d[tau + 1], den = a - 2 * b + c; const sh = den ? (a - c) / (2 * den) : 0; return SR / (tau + sh); }
    return 0;
  }
  // image
  const cw = 160, ch = 90, cv = document.createElement('canvas'); cv.width = cw; cv.height = ch; const g = cv.getContext('2d', { willReadFrequently: true });
  let prevGray = null, prevHist = null, lastScreenT = -1, lastThumbT = -1;
  const A = [], S = [], T = [];
  const adOn = () => player && player.classList.contains('ad-showing');
  function screenSample(t) {
    g.drawImage(v, 0, 0, cw, ch); const px = g.getImageData(0, 0, cw, ch).data; const gray = new Uint8Array(cw * ch); const hist = new Float64Array(64);
    for (let i = 0, j = 0; i < px.length; i += 4, j++) { const y = (px[i] * 0.299 + px[i + 1] * 0.587 + px[i + 2] * 0.114) | 0; gray[j] = y; hist[y >> 2]++; }
    const tot = cw * ch; let bc = 0; if (prevHist) for (let k = 0; k < 64; k++) bc += Math.sqrt(hist[k] / tot * prevHist[k] / tot);
    let mot = 0; if (prevGray) { for (let j = 0; j < tot; j++) mot += Math.abs(gray[j] - prevGray[j]); mot /= tot; }
    let edge = 0, ne = 0; for (let y = Math.floor(ch * 0.75); y < ch - 1; y++) for (let x = 0; x < cw - 1; x++) { const j = y * cw + x; const gx = gray[j + 1] - gray[j], gy = gray[j + cw] - gray[j]; if (gx * gx + gy * gy > 900) edge++; ne++; }
    S.push([+t.toFixed(2), prevHist ? +Math.sqrt(Math.max(0, 1 - bc)).toFixed(3) : 0, +mot.toFixed(2), +(edge / ne).toFixed(4)]);
    prevHist = hist; prevGray = gray;
    if (t - lastThumbT >= 1) { lastThumbT = t; T.push([+t.toFixed(2), cv.toDataURL('image/jpeg', 0.6).slice(23)]); }
  }
  proc.onaudioprocess = (e) => {
    if (v.paused || v.ended || adOn()) return;
    if (!started) { started = true; if (v.currentTime > 0.5) { v.currentTime = 0; return; } }
    const t = v.currentTime; const x = e.inputBuffer.getChannelData(0);
    let ss = 0; for (let i = 0; i < N; i++) { ss += x[i] * x[i]; re[i] = x[i] * win[i]; im[i] = 0; }
    const rms = Math.sqrt(ss / N); fft();
    let my = 0; const ys = []; for (let k = lo; k <= hi; k++) { const yv = 20 * Math.log10(Math.hypot(re[k], im[k]) + 1e-9); ys.push(yv); my += yv; } my /= ys.length;
    let cov = 0; for (let i = 0; i < ys.length; i++) cov += (fk[i] - fmean) * (ys[i] - my);
    const f0 = rms > 0.003 ? yin(x) : 0;
    A.push([+t.toFixed(3), +(20 * Math.log10(rms + 1e-9)).toFixed(1), +f0.toFixed(1), +(cov / fvar).toFixed(2)]);
    if (t - lastScreenT >= 0.2) { lastScreenT = t; try { screenSample(t); } catch (err) { } }
  };
  window.__VIS = { A, S, T, done: false, name: NAME, duration: v.duration };
  v.addEventListener('ended', () => { window.__VIS.done = true; });
  await ctx.resume(); await v.play();
  return 'started ' + NAME + ' ' + v.duration;
})();
