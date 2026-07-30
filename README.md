# SMART: Semi-supervised Multi-Action Recognition via Transductive graph propagation

<!-- TODO: confirm or replace the expansion above -->

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-orange)](https://pytorch.org/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey)](LICENSE)

> **&lt;Author One, Author Two, Author Three&gt;**
> &lt;Affiliation 1&gt; · &lt;Affiliation 2&gt;
> &lt;Venue Year&gt; — Paper ID &lt;NNN&gt;

---

## Overview

Most action-recognition pipelines assume one label per frame and dense
annotation. Real egocentric and home-activity video breaks both assumptions: a
person sits on a sofa *while* holding a book *while* reading it, and labelling
every frame of every video is not affordable.

SMART recovers **all concurrent actions on every frame from roughly 10% labelled
frames**, with no per-frame classifier trained on the full data. It has two
parts:

**1 — A focused multimodal feature front-end.**
Each frame passes through **DEFT** (Deformable Egocentric Focus Transform): a
bounded spatial transformer plus a fixed Gaussian centre prior that pulls the
frozen image encoder onto the hand/object region carrying the action instead of
the wall the camera happens to face. RGB and optical-flow embeddings are then
combined by **cross-modal co-attention**, in which a per-sample agreement scalar
gates how much each stream injects into the other. DEFT and the fusion are
trained only on the ~10% seed frames, supervised through a throwaway linear head
under **Asymmetric Loss**; the head is discarded afterwards.

**2 — Transductive propagation over a per-window graph.**
Frames within an N-frame window become nodes of a similarity graph: a Gaussian
**PDF** kernel over cosine distance with a data-driven bandwidth, pruned by a
**CDF** rule that keeps, per frame, only the strongest neighbours holding
`gamma` of its probability mass, plus a temporal chain so no frame is orphaned.
Labels then diffuse from the seeds by **one independent random walk per class**.

That independence is the crux. Classical label propagation renormalises scores
across classes at every step, so classes compete and the second concurrent
action is suppressed by construction. SMART's walks never compete, so two
co-occurring actions can both reach high scores on the same frame.

A **layered head** turns the raw walk scores into label sets: stationary
subtraction (remove the graph's structural prior), per-class calibration,
temporal smoothing, a co-occurrence boost estimated from the seeds, and
**per-frame row centering** — subtracting each frame's mean score across classes,
which removes the shared "how much is happening here" component and leaves only
the which-action signal. Sets are decoded with per-class F-beta thresholds fitted
on the seeds.

![Problem formulation](images/SMART_Problem_Formulation.pdf)

![Architecture](images/SMART_Architecture.pdf)

---

## Results

Lead metric is **mAP** over classes present in the evaluated frames. Seed frames
are excluded from evaluation. Protocol: per-unit, then mean across units
(see [`Splits/README.txt`](Splits/README.txt)).

### ADL

| Method | mAP | macro-F1 | micro-F1 | top-1 |
|---|---|---|---|---|
| &lt;baseline&gt; | — | — | — | — |
| &lt;baseline&gt; | — | — | — | — |
| **SMART (ours)** | **—** | **—** | **—** | **—** |

### Charades

| Method | mAP | macro-F1 | micro-F1 | top-1 |
|---|---|---|---|---|
| &lt;baseline&gt; | — | — | — | — |
| &lt;baseline&gt; | — | — | — | — |
| **SMART (ours)** | **—** | **—** | **—** | **—** |

> Numbers depend on a random 10% seed draw. Section 8 of
> [`tools/ablations.ipynb`](tools/ablations.ipynb) re-draws the seeds five times
> and reports mean ± std with a 95% CI, which is the number to quote.

### Ablations

| Configuration | ADL mAP | Charades mAP |
|---|---|---|
| **SMART (full)** | **—** | **—** |
| − row centering | — | — |
| − CDF sparsification | — | — |
| − temporal edges | — | — |
| − co-occurrence boost | — | — |
| joint propagation (classes compete) instead of independent walks | — | — |

Reproduce all of these with sections 2–3 of `tools/ablations.ipynb`, or from the
command line:

```bash
python evaluate.py --data_dir artifacts/adl --feat_suffix _resnet --window 150 --no_row_center
python evaluate.py --data_dir artifacts/adl --feat_suffix _resnet --window 150 --prop_mode joint
```

---

## Repository Structure

```
SMART/
├── dataset.py              # Step 1: class map, multi-hot targets, windows, seeds
├── train.py                # Step 2: train DEFT + fusion on the 10% seeds
├── extract_features.py     # Step 3: extract per-frame features
├── evaluate.py             # Step 4: graph -> random walk -> head -> mAP/F1
├── model.py                # DEFT, CoAttnFusion, ASL, backbone factory
├── requirements.txt
├── LICENSE
├── tools/
│   ├── imports.py          # common imports, seeding, sanity-check helper
│   ├── graph.py            # PDF, CDF/kNN pruning, temporal edges, random walks
│   ├── head.py             # smoothing, co-occurrence, thresholds, decoders
│   ├── metrics.py          # mAP, micro/macro F1, top-k, LRAP
│   ├── ablations.ipynb     # variant tables, sweeps, significance testing
│   └── visualize_graph.ipynb  # Plotly random-walk visualisation
├── ADL/Labels/             # per-participant annotation CSVs
├── Charades/Labels/        # the single Charades annotation CSV
├── Splits/                 # seed-mask protocol (README.txt)
└── images/                 # architecture and qualitative figures
```

Generated at runtime and **deliberately not tracked** (see `.gitignore`):
`artifacts/` (targets, seeds, features) and `checkpoints/` (weights). These are
distributed through GitHub Releases instead — a single ADL feature file is
several megabytes and committing them makes the repository impractical to clone.

---

## Installation

### Requirements

- Python **3.10** or higher
- CUDA-compatible GPU recommended (CPU works but feature extraction is slow)
- 16 GB+ RAM

### Step 1 — Clone

```bash
git clone https://github.com/pawanesh-mnnit/SMART.git
cd SMART
```

### Step 2 — Virtual environment

```bash
python -m venv smart_env

# Linux / macOS
source smart_env/bin/activate

# Windows (Command Prompt)
smart_env\Scripts\activate

# Windows (Git Bash)
source smart_env/Scripts/activate
```

### Step 3 — Install PyTorch

Check your CUDA version first:

```bash
nvidia-smi
```

Then install the matching build:

```bash
# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Step 4 — Remaining dependencies

```bash
pip install -r requirements.txt
```

### Step 5 — Verify

```bash
python -c "
import torch, torchvision, numpy, pandas, scipy, matplotlib, networkx, plotly
print('torch      :', torch.__version__)
print('torchvision:', torchvision.__version__)
print('CUDA       :', torch.cuda.is_available())
print('All OK')
"
```

---

## Quick Sanity Check (No Dataset Required)

Verifies the graph, the random walk and the head end to end on synthetic data,
with no dataset and no checkpoint. It should finish in a couple of seconds.

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
cleaner than real video, so the mAP here means nothing beyond "the tensors flow".
What it does confirm is that `P` is column-stochastic, that the shapes survive
the whole chain, and that the head emits genuine multi-label sets rather than
collapsing to one class per frame. For the actual comparison against joint
propagation, run section 3 of [`tools/ablations.ipynb`](tools/ablations.ipynb) on
real features.

---

## Datasets

### ADL

Download from <https://www.csee.umbc.edu/~hpirsiav/papers/ADLdataset/>.

```
/path/to/ADL/
├── Frames/
│   ├── P_09/Original_P_09/frame_00000.jpg   (5-digit, 0-indexed)
│   └── ...
├── OpticalFlow/
│   ├── P_09/viz/frame_00000.jpg
│   └── ...
└── Label/
    ├── P_09_labeled.csv                     (copy these into ADL/Labels/)
    └── ...
```

### Charades

Download from <https://prior.allenai.org/projects/charades>.

```
/path/to/Charades/
├── RGB/
│   ├── 00HFP/rgb/frame_000001.jpg           (6-digit, 1-indexed)
│   └── ...
├── Flow/
│   ├── 00HFP/flow/frame_000001.jpg
│   └── ...
└── Label/
    └── Charades_Annotation.csv              (copy into Charades/Labels/)
```

Annotation schemas are documented in
[`ADL/Labels/README.md`](ADL/Labels/README.md) and
[`Charades/Labels/README.md`](Charades/Labels/README.md).

---

## Pretrained Models

Download checkpoints and pre-extracted features from the GitHub Release, then
skip straight to Step 4:

**[Download from GitHub Releases v1.0.0](https://github.com/pawanesh-mnnit/SMART/releases/tag/v1.0.0)**

| File | Type | Dataset | Configuration |
|---|---|---|---|
| `adl_deft_fusion_resnet.pt` | checkpoint | ADL | ResNet-50 + flow fusion |
| `adl_deft_fusion_effnet.pt` | checkpoint | ADL | EfficientNet-B0 + flow fusion |
| `adl_deft_fusion_clip.pt` | checkpoint | ADL | CLIP ViT-B/32 + flow fusion |
| `adl_deft_rgbonly.pt` | checkpoint | ADL | RGB only (ablation) |
| `adl_deft_flowonly.pt` | checkpoint | ADL | flow only (ablation) |
| `charades_deft_fusion_resnet.pt` | checkpoint | Charades | ResNet-50 + flow fusion |
| `adl_artifacts.zip` | features + targets | ADL | all units, all suffixes |
| `charades_artifacts.zip` | features + targets | Charades | all units, all suffixes |

```bash
mkdir -p checkpoints artifacts
unzip adl_artifacts.zip -d artifacts/adl
# then run Step 4 below
```

---

## Usage

### Step 1 — Build targets, windows and seeds

```bash
# ADL
python dataset.py --dataset adl \
    --rgb_root  /path/to/ADL/Frames \
    --anno_root ADL/Labels \
    --out_dir   artifacts/adl \
    --units P_09 P_10 P_11 P_12 P_16 P_17 \
    --window 100 --seed_frac 0.10 --seed 42

# Charades
python dataset.py --dataset charades \
    --rgb_root /path/to/Charades/RGB \
    --anno_csv Charades/Labels/Charades_Annotation.csv \
    --out_dir  artifacts/charades \
    --window 60 --seed_frac 0.10 --seed 42
```

Run this over **all** units at once — the class map must be shared for pooled
evaluation and for cross-unit co-occurrence to mean anything.

### Step 2 — Train DEFT + fusion on the seeds

```bash
python train.py --dataset adl \
    --rgb_root  /path/to/ADL/Frames \
    --flow_root /path/to/ADL/OpticalFlow \
    --data_dir  artifacts/adl \
    --backbone  resnet50 \
    --epochs 3 --batch 32 \
    --save_path checkpoints/adl_deft_fusion_resnet.pt
```

**Keep the epoch count small.** There are only a few hundred seed frames; 3
epochs is right for a frozen encoder. Training until the seed loss bottoms out
memorises the seeds and the propagated scores get *worse*. Use `--epochs 15
--finetune_block layer4` only when deliberately fine-tuning a ResNet block.

### Step 3 — Extract per-frame features

```bash
python extract_features.py \
    --checkpoint checkpoints/adl_deft_fusion_resnet.pt \
    --rgb_root   /path/to/ADL/Frames \
    --flow_root  /path/to/ADL/OpticalFlow \
    --data_dir   artifacts/adl
```

Writes `artifacts/adl/<unit>_feats_resnet.npz`. The suffix encodes the
configuration, so several feature sets coexist in one folder and the feature
ablations are just a different `--feat_suffix` at Step 4.

### Step 4 — Propagate and evaluate

```bash
python evaluate.py \
    --data_dir artifacts/adl \
    --feat_suffix _resnet \
    --units P_09 P_10 P_11 P_12 P_16 P_17 \
    --window 150 --rw_steps 10 --gamma 0.90
```

Useful additions:

```bash
--protocol pooled       # global thresholds and co-occurrence over pooled seeds
--diagnostics           # zero-seed windows, sigma spread, per-class AP
--save_labels           # per-frame predicted label SETS as CSV
--timeline P_11         # qualitative GT-vs-SMART figure into images/
--save_json results.json
```

---

## Hyperparameters

| Parameter | ADL | Charades | Flag |
|---|---|---|---|
| Window size N | 100 (150 headline) | 60 | `--window` |
| Seed fraction | 0.10 | 0.10 | `--seed_frac` |
| Min seeds per class | 2 | 2 | `--min_per_class` |
| Random seed | 42 | 42 | `--seed` |
| Feature normalisation | center | center | `--feat_norm` |
| Kernel bandwidth σ | median pairwise distance | median | `--sigma_mode` |
| Sparsification | CDF | CDF | `--sparsify` |
| CDF threshold γ | 0.90 | 0.90 | `--gamma` |
| kNN neighbours (baseline) | 10 | 10 | `--knn_k` |
| Random-walk steps | 10 | 10 | `--rw_steps` |
| Temporal smoothing window | 5 | 5 | `--w_smooth` |
| Co-occurrence weight λ | 0.3 | 0.3 | `--lam` |
| Row centering | on | on | `--no_row_center` to disable |
| Decoder | F-beta thresholds | F-beta | `--decoder` |
| β (threshold fitting) | 0.5 | 0.5 | `--fbeta` |
| Max labels per frame | 3 | 3 | `--max_k` |
| Background gate percentile | 10 | 10 | `--act_pctl` |
| **Front-end** | | | |
| Backbone | ResNet-50 (frozen) | ResNet-50 | `--backbone` |
| Image size | 224 | 224 | `--img_size` |
| DEFT Gaussian σ_g | 0.5 | 0.5 | — |
| DEFT affine bound | ±0.3 scale/shear, ±0.5 translation | same | — |
| Loss | Asymmetric (γ⁻=4, γ⁺=1, clip=0.05) | same | `--loss` |
| Epochs (seeds only) | 3 | 3 | `--epochs` |
| Batch size | 32 | 32 | `--batch` |
| LR — fusion + head | 1e-3 | 1e-3 | `--lr` |
| LR — DEFT | 5e-3 | 5e-3 | `--deft_lr` |
| LR — unfrozen backbone block | 1e-4 | 1e-4 | `--backbone_lr` |
| Weight decay | 1e-4 | 1e-4 | `--weight_decay` |
| Optimizer | Adam | Adam | — |

DEFT gets the highest learning rate because its gradient is the weakest in the
chain — it arrives through a frozen encoder.

---

## Evaluation Protocol

Full detail in [`Splits/README.txt`](Splits/README.txt). In brief:

- Seeds are ~10% of a unit's **labelled** frames, chosen so that every class
  present contributes at least 2 seeds, with the remainder drawn at random.
- Seed frames are **excluded** from evaluation — scoring them would inflate
  every metric.
- `per_unit` (default): each unit runs independently, metrics averaged with
  equal weight per unit.
- `pooled`: per-unit graphs, but global co-occurrence, global thresholds and
  metrics pooled over every non-seed frame.

---

## Model Efficiency

| Component | Parameters | Notes |
|---|---|---|
| DEFT | ~30 K | the only spatial module trained |
| Co-attention fusion | 2·d² + d | 2.1 M at d = 1024 |
| Backbone | frozen | not counted as trainable |
| Propagation + head | **0** | no learned parameters at all |
| **Total trainable** | **—** | fill in for your configuration |

| Stage | Cost |
|---|---|
| Feature extraction | — ms/frame |
| Graph construction (N = 100) | — ms/window |
| Random walk (10 steps, C classes) | — ms/window |
| Head + decode | — ms/window |

*Measured on &lt;GPU&gt;, averaged over 100 runs.*

The propagation stage has **no learned parameters** — it is a sequence of
matrix products. Inference cost is O(N²D) for the graph plus O(t·N²·C) for the
walks, per window.

---

## Tested Environment

| Package | Version |
|---|---|
| Python | 3.10.x |
| torch | 2.5.1+cu121 |
| torchvision | 0.20.1+cu121 |
| numpy | 1.24+ |
| pandas | 2.0+ |
| scipy | 1.10+ |
| networkx | 3.1+ |
| plotly | 5.15+ |
| GPU | &lt;your GPU&gt; |
| CUDA | 12.1 |

---

## Citation

```bibtex
@inproceedings{smart2026,
  title     = {SMART: <full title>},
  author    = {<Author One and Author Two and Author Three>},
  booktitle = {<Venue>},
  year      = {2026}
}
```

---

## License

Released under the [CC BY-NC 4.0 License](LICENSE). Free for academic and
research use; commercial use requires permission. The datasets and pretrained
backbones carry their own licenses.
