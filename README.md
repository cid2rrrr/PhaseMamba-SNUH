# PhaseMamba · RTX 3090 · Conda · INFERENCE ONLY

This package is extracted from the **October 23 `phasemamba_vitrectomy_cv_original_speed_full.zip`** `vitphase.models` implementation. It loads the EXACT corresponding trained `Stage0 / Stage1 / Stage2` tensors without any training. The **Stage2 activity_head is trained binary Inactive=0 / Active=1** and is always evaluated; output has both phase and activity probabilities.

This is the independent paper-derived PhaseMamba reimplementation previously provided for SNUH Vitrectomy, **not** an official author model, and **does not include trained weights**. It is not interchangeable with the separate `phasemamba/model.py` or TeCNO activity head checkpoints.

## Requirements

- Ubuntu 20.04/22.04/24.04 (or WSL2 Ubuntu), RTX 3090 24 GB, NVIDIA driver supporting CUDA 12.1 or later (check `nvidia-smi`).
- Miniconda / Anaconda available on the host, internet during installation, `git` and FFmpeg (installed below).
- **Three sets of existing local files**: (1) your trained Vitphase experiment folder containing `run_config.json`, stage1/stage2 checkpoints, and optionally stage0 and `transition_prior.npy`; (2) the **original** `configs/labels.yaml` from training; (3) the **trusted original** LoViT spatial checkpoint `Trained_VIT_Cholec80.pth`.  When the model was trained with `--skip-stage0` and no replacement Stage0 checkpoint, only the original LoViT file is used for spatial features.

## 1. Create Conda environment (no Docker)

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg git
conda create -n phasemamba-infer python=3.10 -y
conda activate phasemamba-infer
cd PhaseMamba_3090_Conda_Inference
bash scripts/install_3090.sh
```

`install_3090.sh` installs the pinned PyTorch **2.5.1 + CUDA 12.1 runtime**, compatible torchvision, LoViT's required `timm==0.4.12`, `transformers==4.44.2` to avoid the old `GreedySearchDecoderOnlyOutput` issue, then the **precompiled official CUDA 12 / PyTorch 2.5 / CPython 3.10 / ABI-matching** `causal-conv1d==1.5.0.post8` and `mamba-ssm==2.2.4` wheels.  It performs a real RTX3090/CUDA Mamba forward smoke test. No `nvcc`/CUDA toolkit is required for the wheel path; only a sufficiently recent NVIDIA *driver*. Mamba-1 CUDA kernels are mandatory, never substituted by a CPU-only implementation.

If you already have Python 3.10 and PyTorch 2.5.1 CUDA 12.1 in the activated Conda environment, run individual install steps in `scripts/install_3090.sh` as appropriate. Do not mix environments' site-packages.

## 2. Place source and checkpoint files

```text
PhaseMamba_3090_Conda_Inference/
  pm_infer/                      # original Stage0/1/2 architecture + deployment CLI
  configs/labels.yaml           # example from our earlier code; MUST equal the original training labels
  third_party/LoViT/models/     # official LoViT code
  weights/Trained_VIT_Cholec80.pth
  checkpoints/full/
      run_config.json
      stage0/last.pt             # only when original full run fine-tuned Stage0
      stage1/last.pt
      stage2/last.pt             # phase_head AND activity_head weights
      transition_prior.npy
```

To get LoViT source (checkpoint weight is downloadable via authors' official README):

```bash
git clone https://github.com/MRUIL/LoViT.git third_party/LoViT
mkdir -p weights checkpoints/full
```

Official LoViT repository: https://github.com/MRUIL/LoViT

`--run-dir` can also be a `runs/vitrectomy/cv/fold_00` folder; in this case use `--weights best` to select held-out-validation checkpoints. Full-data training uses `--weights last` (no validation split). **CV fold models are not a full-data model** and must be identified as such when evaluating new videos.

`run_config.json` and `labels.yaml` are mandatory to prevent accidentally mixing incompatible class definitions. For previous Vitrectomy training, IDs `0,12,13` were excluded from the **phase classifier** but were used as **Inactive** supervision for the binary head. The class order seen by the model is the ordered `phase_order` from labels.yaml (not necessarily the original numerical IDs).

## 3. Check the environment

```bash
python -m pm_infer.run --check
# After installing the LoViT repository and spatial checkpoint:
python -m pm_infer.run --check --check-lovit
```

Test must report `Mamba CUDA kernel: PASS`, `CUDA runtime=12.1`, `GPU=NVIDIA GeForce RTX 3090` (GPU display name may differ slightly).

## 4. Inference over single video / folder

```bash
# a) Single video, original resolution overlay at 2x playback, both heads on.
python -m pm_infer.run \
  --run-dir checkpoints/full \
  --input /dataset/external/Surgery_9999.mp4 \
  --labels configs/labels.yaml \
  --lovit-repo third_party/LoViT \
  --lovit-checkpoint weights/Trained_VIT_Cholec80.pth \
  --activity-gate \
  --overlay-resolution original \
  --overlay-speed 2 \
  --output outputs/single

# b) Folder (all mp4/mov/mkv/avi), 4x playback, don't overwrite phase on inactive frames.
python -m pm_infer.run \
  --run-dir checkpoints/full \
  --input /dataset/external/mp4 \
  --labels configs/labels.yaml \
  --overlay-speed 4 \
  --output outputs/external

# c) Faster, skip overlay video, keep CSV + timeline PNG and .npz probabilities.
python -m pm_infer.run --run-dir checkpoints/full \
  --input /dataset/external/mp4 --no-overlay --output outputs/fast
```

**Note:** The source project sets `postprocess.activity_gate=false` by default. Even without `--activity-gate` the Active/Inactive head is executed and written to the CSV; the option only changes the `final` phase result to `-1` on predicted Inactive frames. Use `--no-activity-gate` to override a training config that enables gating. For phase-order correction use the saved `transition_prior.npy` or pass `--no-refinement` to use raw phase labels. This deployment does not invoke optional EGTP/StableSPR; it uses the original causal phase-order refinement.

## 5. Inference from previously extracted features (optional)

If you already have LoViT **768-D frame features from the SAME spatial encoder/checkpoint and exactly the same 1 FPS preprocessing**, inference can skip LoViT loading and take a `.pt`:

```bash
python -m pm_infer.run --run-dir checkpoints/full \
  --features-pt /dataset/external/Surgery_9999.pt \
  --labels configs/labels.yaml --no-overlay \
  --activity-gate --output outputs/from_features
```

`.pt` must hold `torch.Tensor[T,768]` or dict `{'features': Tensor[T,768], 'sec': Tensor[T]}`; cached features from an incompatible LoViT fine-tune stage **must not** be reused. Feature input generates CSV/timeline but no video overlay.

## 6. Outputs

```text
outputs/single/
  Surgery_9999/
    predictions.csv       # per original sampled second; predicted phase and activity
    probabilities.npz     # phase (T,C), activity (T,2), seconds (T,)
    timeline.png          # raw / refined / activity / final timeline
    overlay.mp4           # original-resolution annotated 1fps samples, speed multiplier configurable
    decode_info.json
    provenance.json
  summary.json
```

`predictions.csv` fields: `sec, raw, raw_name, refined, refined_name, activity, activity_confidence, p_active, p_inactive, final, final_name, confidence, ...`. `activity` = `0 Inactive` / `1 Active`, and `final=-1` means **Inactive** when gate enabled. 1 FPS temporal sampling is **not** the same as native-FPS frame-wise output. `overlay_speed=4` plays sampled frames at 4 FPS, i.e. approximately 4× speed.

## 7. Troubleshooting

- `size mismatch` / `unexpected keys`: checkpoints come from a different architecture or label mapping. **Do not use `strict=False`** to silence it. These scripts are for `vitphase.models.Stage2` (`activity_head` internal), not the 2026-10-21 separate TeCNO binary head and not `phasemamba/model.py` from the earlier project.
- `GreedySearchDecoderOnlyOutput`: installed transformers version incompatible with the Mamba wheel; run `python -m pip install 'transformers==4.44.2'` then reinstall appropriate Mamba wheel if import still fails.
- `libc10.so` or CUDA undefined-symbol failure: usually wheel mismatched to PyTorch CUDA/ABI/Python. Recreate Conda env and install matching wheels; inspect `python -c 'import torch;print(torch.__version__,torch.version.cuda,torch._C._GLIBCXX_USE_CXX11_ABI)'`.
- `cannot import mamba_ssm` or no CUDA kernels: run `python -m pm_infer.run --check` before inference; Mamba must really run on CUDA.
- LoViT `GELU approximate`: original code's `repair_legacy_gelu()` restores missing PyTorch attribute. 224×224 resolution is enforced.
- OOM: lower `--batch-size` from 4 to 1, and `--local-batch-size` from 16 to 4. Stage2 deliberately processes the whole video to preserve Mamba temporal history (does **not** silently chop into independent chunks).
- H.264 broken NAL warnings: FFmpeg salvage/logging is integrated; `--min-coverage 0.8` by default checks recovered one-FPS timestamp coverage. `.log` files are saved with extracted frames.
- `Checkpoint label configuration differs`: replace `configs/labels.yaml` with the exact one from your run. Do not change it just to bypass validation.
- LoViT checkpoint loading uses `weights_only=False` because the original authors distributed a serialized `nn.Module`. **Only load trusted checkpoint files.**

### Testing limitations

The package was syntax-checked and tested with a mocked Mamba network to validate checkpoint semantics and end-to-end CSV export in this environment. We cannot execute CUDA 3090 kernels or the LoViT checkpoint here: the final GPU smoke test must be run on the target machine.
