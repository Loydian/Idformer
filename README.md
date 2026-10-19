# A Simple but Effective Vision Transformer Framework for Visible-infrared Person Re-identification

English | [中文](README_zh.md)

Basic training recipe of Idformer on SYSU-MM01 and RegDB.

## Requirements

Python 3.9, PyTorch 1.12.1 (CUDA 11.6) and one 24 GB GPU (tested on an NVIDIA RTX 4090).

```bash
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```

## Data

```
SYSU-MM01/
  cam1/ ... cam6/
  exp/train_id.txt  exp/val_id.txt  exp/test_id.txt
RegDB/
  Visible/  Thermal/
  idx/{train,test}_{visible,thermal}_{1..10}.txt
```

The RegDB index files follow [Cross-Modal-Re-ID-baseline](https://github.com/mangye16/Cross-Modal-Re-ID-baseline).

## Pre-trained backbone

[swinv2_base_patch4_window12_192_22k.pth](https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_base_patch4_window12_192_22k.pth)
(SwinV2-B, ImageNet-22K).

## Training

SYSU-MM01:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train_sysu.sh /path/to/SYSU-MM01 /path/to/swinv2_base_patch4_window12_192_22k.pth logs
```

RegDB, for each split `TRIAL` (1–10), fine-tuned from a SYSU-MM01 checkpoint or trained from ImageNet-22K weights
(`none`):

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train_regdb.sh /path/to/RegDB /path/to/swinv2_base_patch4_window12_192_22k.pth logs 1 \
    logs/sysu/<run>/checkpoint_max_mAP.pth
CUDA_VISIBLE_DEVICES=0 bash scripts/train_regdb.sh /path/to/RegDB /path/to/swinv2_base_patch4_window12_192_22k.pth logs 1 none
```

Additional arguments are passed to `main.py` and override the script defaults. Logs and
checkpoints are written to `logs/<dataset>/<configuration>/`.

## Evaluation

The model is evaluated after every epoch. To evaluate a saved checkpoint, append `--test-only --resume <checkpoint>`:

```bash
bash scripts/train_sysu.sh /path/to/SYSU-MM01 /path/to/swinv2_base_patch4_window12_192_22k.pth logs \
    --test-only --resume logs/sysu/<run>/checkpoint_max_mAP.pth
bash scripts/train_regdb.sh /path/to/RegDB /path/to/swinv2_base_patch4_window12_192_22k.pth logs 1 none \
    --test-only --resume logs/regdb/<run>/checkpoint_max_mAP.pth
```

- SYSU-MM01: all-search, single-shot.
- RegDB: `--test-mode 2 1` (default) for visible → thermal, `--test-mode 1 2` for thermal → visible.

## Acknowledgements

This code builds on [Swin Transformer](https://github.com/microsoft/Swin-Transformer),
[TransReID](https://github.com/damo-cv/TransReID),
[Cross-Modal-Re-ID-baseline](https://github.com/mangye16/Cross-Modal-Re-ID-baseline) and
[torchvision](https://github.com/pytorch/vision).
