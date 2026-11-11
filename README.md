# Pause, Cut, Call: The Temporal Body of Direct Address

Code, measured values and settings for the exposition "Pause, Cut, Call: The Temporal Body of Direct Address" by Noh Seung Ju (VIS Nordic Journal for Artistic Research, issue 18, "ON / OFF SCREEN").

The exposition contains only sound and image resynthesised from measured values. The original videos and audio were used only for measurement and are not included here. With access to the same three videos, anyone can measure them again with `analyzer.js` and remake every layer with the scripts below. With the measurement files included here, the sound layer, the video layer and the generative algorithm can be remade without the originals.

## Materials

| key | material | duration |
|---|---|---|
| `luce_B0539` | Istituto Luce, Giornale Luce B0539 (film code B053901), Mussolini's speech in Bari, 6 September 1934, as published on YouTube | 419.28 s |
| `mattarella_2022` | President Sergio Mattarella, end of year message, 31 December 2022, as published on YouTube | 1002.56 s |
| `meloni_20230129` | Giorgia Meloni, "Gli appunti di Giorgia", episode on the first hundred days, 29 January 2023, as published on YouTube | 1004.52 s |

The Luce material is subject to the rights notice of Istituto Luce Cinecittà (All rights reserved). Nothing derived from it in this repository contains its sound or image.

## Structure

```
code/
  analyzer.js       in browser measurement on a YouTube watch page (sound every 64 ms, image every 0.2 s)
  convert.py        analyzer output to measurement file; Viola and Jones frontal face cascade (OpenCV)
  luce_content.py   places person markers of the 1934 speech text on the time axis
  operators.py      operators on measured values; noise resynthesis
  plot.py           layout figures
  ladder.py         sound layer, four stages
  video_ladder.py   video layer, four stages
  generative.py     generative algorithm, fixed version
data/
  m_*.json          measurement files: sound, image and content values for each material
  cap_*.json        person markers and speech spans for each material
  luce_text_events.json  token indices of person words in the 1934 speech text and the five anchors
parameters/
  sound_layer_parameters.json, video_layer_parameters.json, generative_parameters.json
```

## Measurement

Sound: decoded by the browser, resampled at 16 kHz, frames of 64 ms. RMS level in dB; f0 by YIN (de Cheveigné and Kawahara 2002) between 60 and 400 Hz; spectral tilt as the slope of the log spectrum between 100 and 5000 Hz, in dB/kHz.

Image: frames reduced to 160×90 pixels, every 0.2 s. A cut is a Hellinger distance above 0.35 between successive 64 bin luminance histograms. Motion is the mean absolute pixel difference. Edge density is measured in the lower quarter of the frame, where subtitles and lettering appear. Once per second, a 160×90 thumbnail is read in greyscale, enlarged three times and searched with the Viola and Jones frontal face cascade (Viola and Jones 2001); the share of the frame occupied by the largest face is kept. Thumbnails are not published.

Content: words referring to "we", "you", "they" and "I" were found in the Italian automatic subtitles (2022, 2023) and in the speech text of the Opera Omnia (1934). Only the time and category of each marker are kept, not the words.

In `m_*.json`, `screen.samples` has the columns `t`, `largest_face_area_fraction`, `lower_quarter_edge_density`, `motion`.

## Remaking the layers

Requirements: Python 3.10 or later, the packages in `requirements.txt`, and `ffmpeg` on the path.

```
pip install -r requirements.txt
python3 code/ladder.py         # writes out/sound_layer/
python3 code/video_ladder.py   # writes out/video_layer/ (needs the sound layer)
python3 code/generative.py     # writes out/generative/ (needs the sound layer)
python3 code/plot.py data/m_luce_B0539.json out/layout_1934_luce "Title" "Subtitle"
```

No step contains a random element whose seed is not fixed. With the measurement files included here, the WAV files of the sound layer and of the generative algorithm, and the decoded video streams of the video layer and of the generative algorithm, are bit identical to those used in the exposition. Compressed audio streams (MP3, AAC) can differ with the encoder version.

### Sound layer

Noise shaped every 64 ms by the measured tilt and level; f0 is never used.

1. level and tilt
2. level: tilt fixed to the median of all three materials, −6.23 dB/kHz
3. presence: any sound, speech or not, at one level
4. speech and pause: sound not produced by the speaker (cheers, applause, music) erased

All files are normalised to −20 dBFS on their sounding parts.

### Video layer

640×360, 10 frames per second, black on white. The area of a black square centred at 40% of the height is the face share times the canvas area; the thickness of a bottom band is the edge density times 120 px; displacement is motion times 0.25 px; each cut inverts the frame for 0.2 s. Stages 2, 3 and 4 erase motion, then the band, then the face. Each video carries the sound layer of the same stage.

### Generative algorithm

Vertical 720×1280, 25 frames per second, 419.28 s, the duration of the 1934 material. The 2022 and 2023 materials are cut into three segments of 419.28 s each; with the 1934 material, seven layers start together at 0 s. Face squares share one centre and are combined by XOR; a cut in any layer inverts the whole frame for 0.2 s; person markers appear for 0.4 s as horizontal lines at 18% (I), 30% (we), 70% (you) and 82% (they) of the height. The sound is the sum of stage 1 of the sound layer of each segment, at 0 dB (1934) and −3 dB (2022, 2023). The settings are in `generative.py` (`P`) and in `parameters/generative_parameters.json`; changing them gives a different version.

## Citation

See `CITATION.cff`. Please cite the archived version with its DOI.

## Licence

Code: MIT licence (`LICENSE`). Measured values, settings and documentation: Creative Commons Attribution NonCommercial NoDerivatives 4.0 International (CC BY-NC-ND 4.0), as for the exposition.

## References

de Cheveigné, Alain, and Hideki Kawahara. "YIN, a Fundamental Frequency Estimator for Speech and Music." *Journal of the Acoustical Society of America* 111, no. 4 (2002): 1917~1930.

Viola, Paul, and Michael Jones. "Rapid Object Detection Using a Boosted Cascade of Simple Features." In *Proceedings of the 2001 IEEE Computer Society Conference on Computer Vision and Pattern Recognition*, vol. 1. Kauai: IEEE, 2001.
