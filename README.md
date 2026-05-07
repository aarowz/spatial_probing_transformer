![Spatial Probing Attention (step 200)](assets/attn_step_200.png)

(note: transformer identifies "red" despite the orange block due to label space being fixed to 5 classes: white, red, green, blue, yellow)

# Spatial-Probing Transformer (PoC)

Custom scaled dot-product multi-head attention, patch + 2D sinusoidal embeddings, a small **SpatialProber** (image self-attention + coordinate cross-attention), synthetic point-probe data, training, and attention visualization.

## Layout

- `spatial_probing_transformer/` — importable Python package (model, embeddings, blocks, prober, data, train, vis).
- `scripts/train.py` — thin entrypoint for training.
- `tests/` — pytest smoke tests.
- `outputs/` — generated artifacts (`attn.pt`, `plots/*.png`); tracked via `.gitkeep` only.
- `.venv/` — local virtual environment (gitignored). Create it once per clone; see below.

## Quick start

Do this **once** after cloning (from the **repository root** — the folder that contains `pyproject.toml`):

```bash
cd spatial_probing_transformer

python3 -m venv .venv
source .venv/bin/activate          # Windows (cmd): .venv\Scripts\activate.bat
                                   # Windows (PowerShell): .venv\Scripts\Activate.ps1

pip install -U pip
pip install -e ".[dev]"
```

That installs the package in **editable** mode plus dev tools (`pytest`), with dependencies from `pyproject.toml` (`torch`, `matplotlib`).

### Every new terminal session

Activate the same environment, then run commands with **`python`** (points at the venv’s interpreter):

```bash
cd spatial_probing_transformer
source .venv/bin/activate          # Windows: see paths above
```

### Train

```bash
python scripts/train.py
```

Same thing without the script wrapper:

```bash
python -m spatial_probing_transformer.train
```

Training writes **`outputs/attn.pt`** and attention PNGs under **`outputs/plots/`** (for example `attn_step_250.png`, `attn_final.png`).

### Tests

```bash
python -m pytest tests/
```

### Without activating (optional)

You can always invoke the venv interpreter explicitly:

```bash
.venv/bin/python scripts/train.py
.venv/bin/python -m pytest tests/
```

### Troubleshooting

- **`ModuleNotFoundError: spatial_probing_transformer`**: run `pip install -e ".[dev]"` **with the venv activated** (or use `.venv/bin/pip install -e ".[dev]"` once).
- **Wrong Python**: confirm `which python` (macOS/Linux) or `where python` (Windows) resolves inside `.venv` after `activate`.

## Run on Explorer (Northeastern HPC)

Use **scratch** for the conda environment and package caches (not `~/`) to avoid home-directory quota errors. Install **`numpy<2`** and **matplotlib** from conda so extensions match; base Anaconda can ship NumPy 2.x and break `matplotlib` with `ImportError: numpy.core.multiarray failed to import`.

### A. One-time rsync from your Mac (repo root)

```bash
cd /Users/az/Documents/GitHub/spatial_probing_transformer
rsync -avz --delete \
  --exclude='.venv/' --exclude='.git/' --exclude='outputs/' \
  --exclude='__pycache__/' --exclude='*.pyc' --exclude='.pytest_cache/' \
  --exclude='*.egg-info/' --exclude='.DS_Store' \
  ./ zhou.aa@login.explorer.northeastern.edu:/scratch/zhou.aa/spatial_probing_transformer/
```

### B. SSH and request a GPU

```bash
ssh zhou.aa@login.explorer.northeastern.edu
job-assist   # e.g. interactive -> gpu -> v100-sxm2 -> 1 node, 1 task, 4 cpus, 08:00:00, 1GB
```

### C. One-time: conda environment on `/scratch` (and sanity check)

```bash
module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"

export CONDA_ENVS_PATH=/scratch/zhou.aa/conda-envs
export CONDA_PKGS_DIRS=/scratch/zhou.aa/conda-pkgs
export CONDA_TMPDIR=/scratch/zhou.aa/conda-tmp
export TMPDIR=/scratch/zhou.aa/tmp
export PIP_CACHE_DIR=/scratch/zhou.aa/pip-cache
export MPLCONFIGDIR=/scratch/zhou.aa/mpl-config
mkdir -p /scratch/zhou.aa/{conda-envs,conda-pkgs,conda-tmp,tmp,pip-cache,mpl-config}

rm -rf /scratch/zhou.aa/conda-envs/spt

conda create -y -p /scratch/zhou.aa/conda-envs/spt python=3.11
conda activate /scratch/zhou.aa/conda-envs/spt

conda install -y -c pytorch -c nvidia -c conda-forge \
  pytorch pytorch-cuda=12.1 "numpy<2" matplotlib

cd /scratch/zhou.aa/spatial_probing_transformer
python -m pip install -U pip
python -m pip install -e ".[dev]"

python -c "import sys, numpy, matplotlib, torch; print(sys.executable); print('numpy', numpy.__version__); print('mpl', matplotlib.__version__); print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

Confirm `sys.executable` is under `/scratch/zhou.aa/conda-envs/spt/bin/python` and NumPy is **below 2.0**.

### D. Each session: activate and train

```bash
module load anaconda3/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
export CONDA_ENVS_PATH=/scratch/zhou.aa/conda-envs
export MPLCONFIGDIR=/scratch/zhou.aa/mpl-config
export TMPDIR=/scratch/zhou.aa/tmp

conda activate /scratch/zhou.aa/conda-envs/spt
cd /scratch/zhou.aa/spatial_probing_transformer
python scripts/train.py
```

Artifacts: `/scratch/zhou.aa/spatial_probing_transformer/outputs/plots/*.png` and `outputs/attn.pt`.

### Pull outputs back to your Mac

```bash
rsync -avz \
  zhou.aa@login.explorer.northeastern.edu:/scratch/zhou.aa/spatial_probing_transformer/outputs/ \
  /Users/az/Documents/GitHub/spatial_probing_transformer/outputs/
```

## Use the library

```python
from spatial_probing_transformer import SpatialProber
```
