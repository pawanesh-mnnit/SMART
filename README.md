# SMART: Semi-Supervised Egocentric Multi-Action Recognition

[![ICVGIP 2026](https://img.shields.io/badge/ICVGIP-2026-blue)](https://icvgip.in/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-orange)](https://pytorch.org/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey)](LICENSE)

> **Pawanesh Kumar Vishwakarma and Abhimanyu Sahu^{*}**
> Department of Computer Science & Engineering,
> Motilal Nehru National Institute of Technology Allahabad, Prayagraj, India
> 


---

## Overview

Egocentric and daily-living videos routinely contain several actions at once — a
person *holds a sandwich* while *eating a sandwich*, then *holds a cup* while
*opening* and *closing a refrigerator*. Existing multi-label methods handle this,
but every one of them needs exhaustive frame-level annotation, which does not
scale to new environments.

SMART predicts **all concurrent actions on every frame from only 10% labelled
frames**. Since so few labels cannot train a reliable classifier directly, SMART
instead exploits the intrinsic structure of video: labels are propagated over a
graph connecting visually and temporally related frames.

![Problem formulation](images/SMART_Problem_Formulation_Diagram.png)

----
## Repository Structure

```
SMART/
├── dataset.py              # Step 1: class map, multi-hot targets, windows, seeds
├── train.py                # Step 2: fine-tune DEFT + fusion on the 10% seeds
├── extract_features.py     # Step 3: extract per-frame fused features
├── evaluate.py             # Step 4: CSVG -> per-class random walk -> head -> mAP
├── model.py                # DEFT, CoAttnFusion, ASL, backbone factory
├── requirements.txt
├── LICENSE
├── tools/
│   ├── imports.py          # common imports, seeding, sanity-check helper
│   ├── graph.py            # CSVG: PDF affinities, CDF/kNN pruning, random walks
│   ├── head.py             # Multi-Action Head: calibration -> row-centering -> decode
│   ├── metrics.py          # mAP, macro/micro F1, Top-1, LRAP
│   ├── ablations.ipynb     # every table above + sweeps + significance testing
│   └── visualize_graph.ipynb  # Figure 5: per-class propagation over the graph
├── ADL/           
├── Charades/     
├── Splits/                 # seed-mask protocol (README.txt)
└── images/                 # figures
```

Generated at runtime and **not tracked** (see `.gitignore`): `artifacts/`
(targets, seeds, features) and `checkpoints/` (weights). These ship through
GitHub Releases instead — a single feature file is several MB.


## Installation

- Python **3.10+**
- CUDA-capable GPU recommended
- 16 GB+ RAM

```bash
git clone https://github.com/pawanesh-mnnit/SMART.git
cd SMART

python -m venv smart_env
source smart_env/bin/activate          # Windows: smart_env\Scripts\activate

# PyTorch matching your CUDA version (check with nvidia-smi)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

Verify:

```bash
python -c "
import torch, torchvision, numpy, pandas, scipy, matplotlib, networkx, plotly
print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())
print('All OK')
"
```

---

## Quick Sanity Check (No Dataset Required)

Runs CSVG → per-class random walk → Multi-Action Head on synthetic data in a
couple of seconds.

```bash
python -c "
import numpy as np
from tools.graph import build_window_graph, propagate
from tools.head import cooccurrence, head_transform, fit_thresholds, h75_threshold
from tools.metrics import evaluate

rng = np.random.default_rng(0)
n, C, D = 120, 5, 64

# ground truth with deliberate overlaps: classes 0 and 1 co-occur on frames 40-70
Y = np.zeros((n, C), np.float32)
Y[0:70, 0] = 1; Y[40:100, 1] = 1; Y[10:30, 2] = 1; Y[75:110, 3] = 1; Y[100:120, 4] = 1
proto = rng.normal(size=(C, D))
F = np.stack([proto[np.where(Y[i] > 0)[0]].mean(0) + 0.25 * rng.normal(size=D)
              if Y[i].sum() else rng.normal(size=D) for i in range(n)])

seeds = np.zeros(n, bool)
for c in range(C):
    seeds[rng.choice(np.where(Y[:, c] > 0)[0], 3, replace=False)] = True

P, sigma = build_window_graph(F)
Co   = cooccurrence(Y[seeds])
raw  = propagate(P, Y, seeds, steps=10)
S    = head_transform(raw, P, Co)
taus = fit_thresholds(S[seeds], Y[seeds])
ev   = ~seeds
pred = h75_threshold(S[ev], taus)
m    = evaluate(pred, S[ev], Y[ev])

print('transition matrix  :', P.shape, '| column-stochastic:', np.allclose(P.sum(0), 1))
print('sigma (data-driven):', round(sigma, 3))
print('raw walk scores    :', raw.shape)
print('head scores        :', S.shape)
print('decoded label sets :', pred.shape)
print('seeds              :', int(seeds.sum()), '/', n, 'frames')
print('mAP                :', round(m['mAP'], 4))
print('multi-action frames predicted:', int((pred.sum(1) >= 2).sum()), '/', len(pred))
print('Pipeline OK')
"
```

Expected output:

```
transition matrix  : (120, 120) | column-stochastic: True
sigma (data-driven): 1.098
raw walk scores    : (120, 5)
head scores        : (120, 5)
decoded label sets : (105, 5)
seeds              : 15 / 120 frames
mAP                : 0.6743
multi-action frames predicted: 80 / 105
Pipeline OK
```

This is a wiring check, not a quality claim — the synthetic clusters are far
cleaner than real video. What it confirms is that `P` is column-stochastic, the
shapes survive the whole chain, and the head emits genuine multi-label sets
rather than collapsing to one class per frame.

---

## Datasets

### Charades

Sigurdsson & Gupta, ECCV 2016 — <https://prior.allenai.org/projects/charades>

```
/path/to/Charades/
├── RGB/00HFP/rgb/frame_000001.jpg         6-digit, 1-indexed
├── Flow/00HFP/flow/frame_000001.jpg
└── Label/Charades_Annotation.csv          copy into Charades/Labels/
```

### ADL

Pirsiavash & Ramanan, CVPR 2012 —
<https://www.csee.umbc.edu/~hpirsiav/papers/ADLdataset/>

```
/path/to/ADL/
├── Frames/P_09/Original_P_09/frame_00000.jpg    5-digit, 0-indexed
├── OpticalFlow/P_09/viz/frame_00000.jpg
└── Label/P_09_labeled.csv                       copy into ADL/Labels/
```

Annotation schemas: [`ADL/Labels/README.md`](ADL/Labels/README.md),
[`Charades/Labels/README.md`](Charades/Labels/README.md).

---

## Pretrained Models

**[Download from GitHub Releases v1.0.0](https://github.com/pawanesh-mnnit/SMART/releases/tag/v1.0.0)**

| File | Type | Dataset | Configuration |
|---|---|---|---|
| `charades_deft_fusion_effnet.pt` | checkpoint | Charades | **main result** |
| `adl_deft_fusion_effnet.pt` | checkpoint | ADL | **main result** |
| `*_deft_fusion_resnet.pt` | checkpoint | both | ResNet-50 ablation |
| `*_deft_fusion_clip.pt` | checkpoint | both | CLIP ablation |
| `*_deft_rgbonly.pt` / `*_deft_flowonly.pt` | checkpoint | both | modality ablation |
| `charades_artifacts.zip` | features + targets | Charades | all suffixes |
| `adl_artifacts.zip` | features + targets | ADL | all suffixes |

```bash
mkdir -p checkpoints artifacts
unzip charades_artifacts.zip -d artifacts/charades
python evaluate.py --data_dir artifacts/charades --feat_suffix _effnet --window 100
```

---

## Usage

### Step 1 — Targets, windows and seeds

```bash
# Charades
python dataset.py --dataset charades \
    --rgb_root /path/to/Charades/RGB \
    --anno_csv Charades/Labels/Charades_Annotation.csv \
    --out_dir  artifacts/charades \
    --window 100 --seed_frac 0.10 --min_per_class 2 --seed 42

# ADL
python dataset.py --dataset adl \
    --rgb_root  /path/to/ADL/Frames \
    --anno_root ADL/Labels \
    --out_dir   artifacts/adl \
    --window 100 --seed_frac 0.10 --min_per_class 2 --seed 42
```

Run this over **all** units at once — the class map must be shared.

### Step 2 — Fine-tune DEFT + fusion on the seeds

```bash
python train.py --dataset charades \
    --rgb_root  /path/to/Charades/RGB \
    --flow_root /path/to/Charades/Flow \
    --data_dir  artifacts/charades \
    --backbone efficientnet_b0 --epochs 3 --batch 32 \
    --save_path checkpoints/charades_deft_fusion_effnet.pt
```

**3 epochs, as in the paper.** There are only a few hundred seed frames; training
until the seed loss bottoms out memorises them and the propagated scores get
*worse*. The backbone stays frozen throughout.

### Step 3 — Extract features

```bash
python extract_features.py \
    --checkpoint checkpoints/charades_deft_fusion_effnet.pt \
    --rgb_root   /path/to/Charades/RGB \
    --flow_root  /path/to/Charades/Flow \
    --data_dir   artifacts/charades
```

Writes `artifacts/charades/<video>_feats_effnet.npz`.

### Step 4 — Propagate and evaluate

```bash
python evaluate.py \
    --data_dir artifacts/charades \
    --feat_suffix _effnet \
    --window 100 --rw_steps 10 --gamma 0.90
```

Extras:

```bash
--protocol pooled      # global thresholds and co-occurrence over pooled seeds
--diagnostics          # zero-seed windows, sigma spread, per-class AP
--save_labels          # per-frame predicted label SETS as CSV
--timeline 00HFP       # Figure 4-style GT-vs-SMART timeline into images/
--save_json results.json
```

---


---

## Qualitative Results

![Qualitative results — Charades](images/Qualitative_Result_1.png)

![Qualitative results — ADL](images/Qualitative_Result_2.png)

SMART recovers overlapping actions over time from sparse supervision. As a person
grasps a doorknob, opens a door and operates a vacuum, the actions *holding a
vacuum*, *taking a vacuum from somewhere* and *tidying something on the floor* are
correctly localised over the same intervals; paired *opening*/*closing* actions of
doors and cabinets are accurately detected. Predictions fragment slightly over
long intervals because propagation is window-wise, but onset and duration are
preserved.

**Failure mode.** On Charades video `VID_00X3U`, the object-centric *taking a
blanket from somewhere* is recognised reliably (0.80 AP), while appearance-driven
actions such as *someone is smiling* (0.41 AP) and *someone is running somewhere*
(0.47 AP) do much worse. These lack distinctive object or motion cues, so visually
similar frames cluster tightly in the graph and propagation becomes unreliable.

Generate your own:

```bash
python evaluate.py --data_dir artifacts/adl --feat_suffix _effnet \
    --window 100 --timeline P_11 --fig_dir images
```

The three-panel propagation visualisation (Figure 5 of the paper — initial frame
graph, seed initialisation, propagated labels) is produced by
[`tools/visualize_graph.ipynb`](tools/visualize_graph.ipynb).

---

## Related Work by the Authors

- **DEFT-DPT**: A Lightweight Multimodal Framework for Egocentric Video Action Recognition.
  Vishwakarma, Singh & Sahu, *IEEE TCSVT*, 2026.
  [github.com/pawanesh-mnnit/deft-dpt](https://pawanesh-mnnit.github.io/deft-dpt/)
- **EgoHAnG**: Graph-Enhanced Horizon-Aware Egocentric Action Anticipation.
  Vishwakarma, Chowdhury & Sahu, *ICPR*, 2026.
  [github.com/pawanesh-mnnit/EgoHANG](https://github.com/pawanesh-mnnit/EgoHANG)

Future work extends the multi-action framework here with action anticipation.

---

## Citation

```bibtex
@inproceedings{vishwakarma2026smart,
  title     = {SMART: Semi-Supervised Egocentric Multi-Action Recognition},
  author    = {Vishwakarma, Pawanesh Kumar and Sahu, Abhimanyu},
  booktitle = {Proceedings of the 17th Indian Conference on Computer Vision,
               Graphics and Image Processing (ICVGIP'26)},
  year      = {2026},
  address   = {Kolkata, India},
  publisher = {ACM}
}
```

---

## License

Released under the [CC BY-NC 4.0 License](LICENSE). Free for academic and research
use; commercial use requires permission. The datasets and pretrained backbones
carry their own licenses.
