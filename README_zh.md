# A Simple but Effective Vision Transformer Framework for Visible-infrared Person Re-identification

[English](README.md) | 中文

Idformer 在 SYSU-MM01 和 RegDB 上的基础训练方案（basic recipe）。

## 环境

Python 3.9，PyTorch 1.12.1（CUDA 11.6），一张 24 GB 显存的 GPU（在 NVIDIA RTX 4090 上测试）。

```bash
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```

## 数据

```
SYSU-MM01/
  cam1/ ... cam6/
  exp/train_id.txt  exp/val_id.txt  exp/test_id.txt
RegDB/
  Visible/  Thermal/
  idx/{train,test}_{visible,thermal}_{1..10}.txt
```

RegDB 的索引文件格式与 [Cross-Modal-Re-ID-baseline](https://github.com/mangye16/Cross-Modal-Re-ID-baseline) 相同。

## 预训练主干

[swinv2_base_patch4_window12_192_22k.pth](https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_base_patch4_window12_192_22k.pth)（SwinV2-B，ImageNet-22K）。

## 训练

SYSU-MM01：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train_sysu.sh /path/to/SYSU-MM01 /path/to/swinv2_base_patch4_window12_192_22k.pth logs
```

RegDB：对每个划分 `TRIAL`（1–10），可以从 SYSU-MM01 的 checkpoint 微调，也可以只用 ImageNet-22K 权重训练（填 `none`）：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train_regdb.sh /path/to/RegDB /path/to/swinv2_base_patch4_window12_192_22k.pth logs 1 \
    logs/sysu/<run>/checkpoint_max_mAP.pth
CUDA_VISIBLE_DEVICES=0 bash scripts/train_regdb.sh /path/to/RegDB /path/to/swinv2_base_patch4_window12_192_22k.pth logs 1 none
```

追加的参数会传给 `main.py`，并覆盖脚本中的默认值。日志和 checkpoint 保存在 `logs/<数据集>/<配置名>/` 下。

## 评测

每个 epoch 结束后都会评测。要评测已保存的 checkpoint，在命令最后加上 `--test-only --resume <checkpoint>`：

```bash
bash scripts/train_sysu.sh /path/to/SYSU-MM01 /path/to/swinv2_base_patch4_window12_192_22k.pth logs \
    --test-only --resume logs/sysu/<run>/checkpoint_max_mAP.pth
bash scripts/train_regdb.sh /path/to/RegDB /path/to/swinv2_base_patch4_window12_192_22k.pth logs 1 none \
    --test-only --resume logs/regdb/<run>/checkpoint_max_mAP.pth
```

- SYSU-MM01：all-search，single-shot。
- RegDB：`--test-mode 2 1`（默认）评测可见光 → 红外，`--test-mode 1 2` 评测红外 → 可见光。

## 致谢

本代码基于 [Swin Transformer](https://github.com/microsoft/Swin-Transformer)、[TransReID](https://github.com/damo-cv/TransReID)、[Cross-Modal-Re-ID-baseline](https://github.com/mangye16/Cross-Modal-Re-ID-baseline) 和 [torchvision](https://github.com/pytorch/vision)。
