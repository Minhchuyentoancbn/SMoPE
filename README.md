#  One-Prompt Strikes Back: Sparse Mixture of Experts for Prompt-based Continual Learning (ICLR 2026)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/arXiv-Paper-red)](https://arxiv.org/abs/2509.24483)
[![Code](https://img.shields.io/badge/Code-PyTorch-green)](#)

> Official repository for our paper: **"One-Prompt Strikes Back: Sparse Mixture of Experts for Prompt-based Continual Learning", ICLR 2026**

## Abstract

Prompt-based methods have recently gained prominence in Continual Learning (CL) due to their strong performance and memory efficiency. A prevalent strategy in this paradigm assigns a dedicated subset of prompts to each task, which, while effective, incurs substantial computational overhead and causes memory requirements to scale linearly with the number of tasks. Conversely, approaches employing a single shared prompt across tasks offer greater efficiency but often suffer from degraded performance due to knowledge interference. 

To reconcile this trade-off, we propose **SMoPE**, a novel framework that integrates the benefits of both task-specific and shared prompt strategies. Inspired by recent findings on the relationship between Prefix Tuning and Mixture of Experts (MoE), SMoPE organizes a shared prompt into multiple "prompt experts" within a sparse MoE architecture. For each input, only a select subset of relevant experts is activated, effectively mitigating interference. To facilitate expert selection, we introduce a prompt-attention score aggregation mechanism that computes a unified proxy score for each expert, enabling dynamic and sparse activation. Additionally, we propose an adaptive noise mechanism to encourage balanced expert utilization while preserving knowledge from prior tasks. To further enhance expert specialization, we design a prototype-based loss function that leverages prefix keys as implicit memory representations. 

Extensive experiments across multiple CL benchmarks demonstrate that SMoPE consistently outperforms task-specific prompt methods and achieves performance competitive with state-of-the-art approaches, all while significantly reducing parameter counts and computational costs.

<p align="center">
<img src="https://github.com/Minhchuyentoancbn/SMoPE/blob/master/figures/model.pdf" alt="Alt text" width="400"/>
</p>

## Requirements
Create a conda environment and install the following packages:
 * python=3.8.18
 * torch=2.0.0+cu118
 * torchvision=0.15.1+cu118
 * timm=0.9.12
 * scikit-learn=1.3.2
 * numpy
 * pyaml
 * pillow
 * opencv-python
 * pandas
 * openpyxl (write results to a xlsx file)
or simply run:
```setup
pip install -r requirements.txt
```
 
## Datasets
 * Create a folder `data/`
- [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)
- [Imagenet-R](https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar)
- [CUB-200](https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz)

## Checkpoints
 * Create a folder `pretrained/`
 - [Sup-21K](https://huggingface.co/timm/vit_base_patch16_224.augreg_in21k/blob/main/pytorch_model.bin)
 - [Sup-1K](https://huggingface.co/timm/vit_base_patch16_224.augreg2_in21k_ft_in1k/blob/main/pytorch_model.bin)
 - [iBOT-1k](https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitb_16/checkpoint_teacher.pth)
 - [DINO-1k](https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth)  

## Training
Run the following commands under the project root directory. **The scripts are set up for 1 GPUs**.

```bash
sh experiments/cifar-100.sh
sh experiments/imagenet-r_all.sh
sh experiments/cub-200.sh
```

## Results
Results will be saved in a folder named `outputs/`.


## Reference Codes
[1] [CODA-Prompt](https://github.com/GT-RIPL/CODA-Prompt)

[2] [HiDe-Prompt](https://github.com/thu-ml/HiDe-Prompt)

[3] [VQ-Prompt](https://github.com/jiaolifengmi/VQ-Prompt)