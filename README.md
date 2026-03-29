# Pro-PROTAC

**Prototype-driven meta-learning enables cross-ligase generalization for PROTAC degradation activity prediction**

> *Bioinformatics*, 2025 | [Paper](#) | [Data](#)

---

## Overview

Pro-PROTAC is a prototype-based graph neural network trained under an episodic meta-learning framework for cross-ligase PROTAC degradation activity prediction.

**Key features:**
- Requires only SMILES strings as input — no protein structures or language model embeddings needed
- Episodic meta-training on CRBN compounds enables zero-shot transfer to unseen E3 ligases
- Non-parametric prototype classification: no parameter updates at inference time
- Outperforms PROTAC-STAN and DegradeMaster on cross-ligase transfer benchmarks

<p align="center">
  <img src="figures/model_framework.png" width="800"/>
  <br>
  <em>Overview of the Pro-PROTAC framework. The GNN encoder is meta-trained on CRBN episodes and directly transferred to unseen E3 ligases at inference time via prototype-based classification.</em>
</p>

---

## Repository Structure

```
Pro-PROTAC/
├── data_utils.py     # Molecular graph construction, dataset loading, episode sampling
├── models.py         # GCNEncoder and ProtoNet model definitions
├── metrics.py        # AUROC, AUPRC, and other evaluation metrics
├── train_eval.py     # Main training and evaluation entry point
├── visualize.py      # UMAP visualization of learned embeddings
├── requirements.txt  # Python dependencies
└── splits/           # Pre-generated episodic split JSON files
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/Pro-PROTAC.git
cd Pro-PROTAC

# Create a conda environment (recommended)
conda create -n proprotac python=3.9 -y
conda activate proprotac

# Install PyTorch (adjust CUDA version as needed)
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.0 \
    --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric
pip install torch-geometric==2.3.0

# Install remaining dependencies
pip install -r requirements.txt
```

---

## Data Preparation

Download PROTAC-DB from [https://protacdb.com](https://protacdb.com) and place the following files in the project root:

```
protac_filtered_balanced.csv   # Preprocessed and balanced PROTAC dataset
protac.sdf                     # Molecular structures in SDF format
splits/                        # Episodic split files (provided in this repo)
```

The split files follow the naming convention:
```
splits/crbn_vhl_K2Q3_seed{42,2025,3407}.json          # CRBN→VHL, K=2 Q=3
splits/crbn_vhl_K2Q5_bootstrap_seed{42,2025,3407}.json # CRBN→VHL, K=2 Q=5†
splits/crbn_rareE3_K1Q1_seed{42,2025,3407}.json        # Rare E3, K=1 Q=1
splits/crbn_rareE3_K2Q2_seed{42,2025,3407}.json        # Rare E3, K=2 Q=2
```

---

## Usage

### Training and Evaluation

**CRBN → VHL benchmark (K=2, Q=3):**
```bash
for seed in 42 2025 3407; do
    python train_eval.py \
        --csv   protac_filtered_balanced.csv \
        --sdf   protac.sdf \
        --split splits/crbn_vhl_K2Q3_seed${seed}.json \
        --seed  ${seed} \
        --device cuda \
        --outdir results/vhl_K2Q3_seed${seed}
done
```

**CRBN → VHL benchmark (K=2, Q=5†, bootstrap):**
```bash
for seed in 42 2025 3407; do
    python train_eval.py \
        --csv    protac_filtered_balanced.csv \
        --sdf    protac.sdf \
        --split  splits/crbn_vhl_K2Q5_bootstrap_seed${seed}.json \
        --meta-q 5 \
        --seed   ${seed} \
        --device cuda \
        --outdir results/vhl_K2Q5_seed${seed}
done
```

**Rare E3 ligase benchmark (one-shot):**
```bash
for seed in 42 2025 3407; do
    python train_eval.py \
        --csv    protac_filtered_balanced.csv \
        --sdf    protac.sdf \
        --split  splits/crbn_rareE3_K1Q1_seed${seed}.json \
        --meta-k 1 --meta-q 1 \
        --seed   ${seed} \
        --device cuda \
        --outdir results/rareE3_K1Q1_seed${seed}
done
```

### Aggregate Results Across Seeds

```python
import pandas as pd
import numpy as np

seeds   = [42, 2025, 3407]
metrics = ["AUROC", "AUPRC", "Accuracy", "F1", "BALACC"]

data = {m: [] for m in metrics}
for seed in seeds:
    df = pd.read_csv(f"results/vhl_K2Q3_seed{seed}/proprotac_summary_seed{seed}.csv")
    for m in metrics:
        data[m].append(df[m].values[0])

for m in metrics:
    print(f"{m}: {np.mean(data[m]):.3f} ± {np.std(data[m]):.3f}")
```

### UMAP Visualization

First, train with `--save-encoder` to save the encoder weights:

```bash
python train_eval.py \
    --csv          protac_filtered_balanced.csv \
    --sdf          protac.sdf \
    --split        splits/crbn_vhl_K2Q3_seed42.json \
    --seed         42 \
    --device       cuda \
    --outdir       results/vhl_K2Q3_seed42 \
    --save-encoder results/vhl_K2Q3_seed42/encoder_seed42.pt
```

Then generate the UMAP figure:

```bash
# Install optional dependency
pip install umap-learn

python visualize.py \
    --ckpt results/vhl_K2Q3_seed42/encoder_seed42.pt \
    --csv  protac_filtered_balanced.csv \
    --sdf  protac.sdf \
    --seed 42 \
    --out  figures/umap_embedding.png
```

---

## Key Hyperparameters

| Argument | Default | Description |
|---|---|---|
| `--meta-epochs` | 100 | Number of meta-training epochs |
| `--episodes-per-epoch` | 100 | Episodes sampled per epoch |
| `--meta-k` | 2 | Support samples per class |
| `--meta-q` | 3 | Query samples per class |
| `--encoder-hidden` | 128 | GCN hidden dimension |
| `--encoder-layers` | 3 | Number of GCN layers |
| `--meta-lr` | 1e-3 | Adam learning rate |

---

## Results

### CRBN → VHL Transfer Benchmark

| Method | AUROC | AUPRC |
|---|---|---|
| RF + ECFP4 | 0.770 ± 0.017 | 0.809 ± 0.013 |
| GNN | 0.756 ± 0.011 | 0.812 ± 0.007 |
| PROTAC-STAN | 0.542 ± 0.030 | 0.654 ± 0.025 |
| DegradeMaster | 0.661 ± 0.033 | 0.798 ± 0.035 |
| **Pro-PROTAC** | **0.806 ± 0.012** | **0.854 ± 0.007** |

<p align="center">
  <img src="figures/radar_K2Q3.png" width="480"/>
  <br>
  <em>Radar chart comparison on the CRBN→VHL benchmark (K=2, Q=3). Pro-PROTAC consistently occupies the largest area across all five metrics.</em>
</p>

### Rare E3 Ligase Benchmark (One-shot)

| Method | AUROC | AUPRC |
|---|---|---|
| PROTAC-STAN | 0.543 ± 0.047 | 0.757 ± 0.032 |
| DegradeMaster | 0.419 ± 0.107 | 0.628 ± 0.034 |
| **Pro-PROTAC** | **0.650 ± 0.076** | **0.825 ± 0.038** |

---

## Citation

If you use Pro-PROTAC in your research, please cite:

```bibtex
@article{proprotac2025,
  title   = {Pro-PROTAC: Prototype-driven meta-learning enables cross-ligase
             generalization for PROTAC degradation activity prediction},
  author  = {Author1 and Author2},
  journal = {Bioinformatics},
  year    = {2025},
  doi     = {DOI HERE}
}
```

---

## License

This project is licensed under the MIT License.
