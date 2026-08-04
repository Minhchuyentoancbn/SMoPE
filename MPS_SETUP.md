# SMoPE on Apple Silicon (`MPS`)

This repo was originally written for CUDA training. The local branch `mps-setup`
adds device handling so the code can run on Apple Silicon through PyTorch `MPS`.

## Local Environment

```bash
cd /Users/lujiayi/Desktop/summer/SMoPE
uv python install 3.11
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install --python .venv/bin/python \
  pip setuptools wheel \
  torch torchvision \
  timm==0.9.12 numpy==1.24.4 pyaml pyyaml pillow==10.3.0 \
  opencv-python==4.9.0.80 tqdm six pandas openpyxl scikit-learn==1.3.2
```

Notes:

- `faiss-gpu` is intentionally skipped on macOS Apple Silicon.
- `sup1k` and `sup21k` checkpoints can now fall back to `timm` auto-download if
  the local files under `pretrained/` are missing.

## Quick Sanity Check

```bash
source .venv/bin/activate
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("mps available:", torch.backends.mps.is_available())
PY
```

## Suggested First Run

Use a short CIFAR-100 run first before trying the full paper setting.

```bash
source .venv/bin/activate
python -u run.py \
  --config configs/cifar-100_prompt_smope.yaml \
  --gpuid 0 \
  --repeat 1 \
  --overwrite 1 \
  --learner_type prompt \
  --learner_name OnePrompt \
  --prompt_param 50 5 1e-5 1e-5 0.4 \
  --seeds 0 \
  --crct_epochs 5 \
  --ca_batch_size_ratio 1 \
  --log_dir outputs/cifar-100/mps-smoke
```

The code will choose `MPS` automatically on a MacBook when CUDA is unavailable.
