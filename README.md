<div align="center">

# CRAViM

### Hybrid State-Space Models and Denoising Training<br>for Unpaired Medical Image Synthesis

**MICCAI 2025**

Junming Zhang · Shancheng Jiang  
Sun Yat-sen University

[![Paper](https://img.shields.io/badge/Paper-Open_Access-b31b1b.svg)](https://papers.miccai.org/miccai-2025/paper/1652_paper.pdf)
[![DOI](https://img.shields.io/badge/DOI-10.1007%2F978--3--032--04937--7__23-blue.svg)](https://doi.org/10.1007/978-3-032-04937-7_23)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2-EE4C2C.svg)](https://pytorch.org/get-started/previous-versions/)

**[Overview](#overview) · [Installation](#installation) · [Data](#data-preparation) · [Training](#training) · [Testing](#testing) · [Citation](#citation)**

</div>

Official PyTorch implementation of **CRAViM (Convolutional Residual Attention Vision Mamba)** for unpaired medical image synthesis. This repository provides the model, training code, and bidirectional inference and evaluation scripts.

## Overview

CRAViM combines convolutional feature extraction with Vision Mamba to capture local anatomical details and long-range spatial dependencies. A cycle denoise consistency framework, with transport loss and random denoise loss, guides image synthesis between two domains.

- **Hybrid architecture:** convolutional, residual attention, and visual state-space components.
- **Bidirectional synthesis:** learn mappings from domain A to B and from B to A.
- **Medical imaging evaluation:** T1 ↔ T2 brain MRI synthesis on IXI and BraTS2018 in the paper.

For the method and reported experiments, see the [paper](https://papers.miccai.org/miccai-2025/paper/1652_paper.pdf).

## Repository structure

```text
CRAViM/
├── core/
│   ├── base_dataset.py    # Two-domain image dataset loader
│   ├── html.py            # HTML result pages
│   ├── util.py            # Utilities and Visdom training logger
│   └── visualizer.py      # Image export and visualization
├── models/
│   └── networks.py        # CRAViM generators and multi-scale discriminator
├── train.py               # Training entry point
├── test.py                # Bidirectional inference and SSIM evaluation
└── LICENSE
```

## Installation

### Environment

The core environment recorded for this project is:

| Component | Version / requirement |
| :--- | :--- |
| Platform | Linux x86_64 for the prebuilt wheels below |
| Python | 3.10.8 |
| PyTorch | 2.1.2 with CUDA 12.1 |
| torchvision | 0.16.2, matching PyTorch 2.1.2 |
| causal-conv1d | 1.5.0.post8 |
| mamba-ssm | 2.2.4 |
| Hardware | NVIDIA GPU with a compatible CUDA driver |

### Setup

```bash
git clone https://github.com/jmzhang-cv/CRAViM.git
cd CRAViM

conda create -n cravim python=3.10.8 -y
conda activate cravim

python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install "numpy<2" pillow einops timm tqdm scikit-image visdom dominate "transformers==4.38.2"
```

NumPy is constrained below 2 for the older PyTorch stack; Transformers is pinned for compatibility with that stack when importing Mamba. These additional packages are not a complete lockfile of the original experimental environment.

Install the matching CUDA extensions using the wheels below. Both target **Python 3.10, CUDA 12, PyTorch 2.1, Linux x86_64, and `cxx11abiFALSE`**.

<details>
<summary><strong>Mamba and causal-conv1d installation commands</strong></summary>

```bash
python -m pip install "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.0.post8/causal_conv1d-1.5.0.post8%2Bcu12torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
python -m pip install "https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4%2Bcu12torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
```

</details>

If your Python, PyTorch, CUDA, or ABI differs, use a matching build from the upstream [Mamba](https://github.com/state-spaces/mamba) and [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) projects. The provided scripts use CUDA, and `models/networks.py` imports the compiled `selective_scan_cuda` extension.

Check imports and GPU availability from the repository root:

```bash
python -c "import torch, selective_scan_cuda; from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; from models.networks import CRAVIM; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
```

## Data preparation

The preprocessed datasets used in the study are not distributed in this repository. Obtain [IXI](https://brain-development.org/ixi-dataset/) and [BraTS2018](https://www.med.upenn.edu/sbia/brats2018/data.html) from their providers and follow the applicable access, licensing, and citation requirements. The [Reg-GAN repository](https://github.com/Kid-Liet/Reg-GAN) is an additional reference for preprocessed medical imaging data.

Prepare 2D image slices in the following layout, where A and B denote the two modalities (for example, A = T1 and B = T2):

```text
Datasets/IXI/
├── trainA/
│   ├── 001.png
│   └── ...
├── trainB/
│   ├── 001.png
│   └── ...
├── testA/
│   ├── 001.png
│   └── ...
└── testB/
    ├── 001.png
    └── ...
```

- The loader accepts common image formats such as PNG, JPEG, BMP, and TIFF. Convert NIfTI volumes or NumPy arrays to image slices before use; volumetric preprocessing is not included.
- Prepare **256 × 256 grayscale slices** for the default configuration (`--image_size 256 --input_nc 1`). The loader normalizes pixel values to `[-1, 1]`.
- Training does not require paired images. For meaningful SSIM evaluation, provide corresponding test slices in A and B with matching sorted order and equal counts.
- If `testA/` is absent and `valA/` exists, the test loader uses `valA/` and `valB/` instead.

## Training

Start the Visdom server in a separate terminal with the `cravim` environment activated:

```bash
python -m visdom.server -p 6006
```

Then run single-GPU training from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --dataroot /path/to/Datasets/IXI \
    --exp IXI_CRAVIM \
    --phase train \
    --image_size 256 \
    --input_nc 1 \
    --batch_size 4 \
    --num_epoch 100 \
    --save_ckpt_every 10
```

Replace `/path/to/Datasets/IXI` with your dataset root. View training curves and input images at [localhost:6006](http://localhost:6006), in the `IXI_CRAVIM` Visdom environment.

| Argument | Default | Purpose |
| :--- | :--- | :--- |
| `--exp` | `IXI_CRAVIM` | Experiment name and output subdirectory |
| `--batch_size` | `4` | Training batch size |
| `--num_epoch` | `100` | Number of training epochs |
| `--lr_g` / `--lr_d` | `0.0002` / `0.0001` | Generator / discriminator learning rate |
| `--num_timesteps` | `4` | Number of diffusion timesteps |
| `--save_ckpt_every` | `10` | Generator checkpoint interval in epochs |

Generator checkpoints and epoch preview images are saved under:

```text
results/IXI_CRAVIM/train_info/
├── netG_A2B_100.pth
├── netG_B2A_100.pth
└── predict_epoch_100.png
```

## Testing

Use the same experiment name and an epoch for which **both generator checkpoints** exist:

```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
    --dataroot /path/to/Datasets/IXI \
    --exp IXI_CRAVIM \
    --phase test \
    --epoch_id 100 \
    --image_size 256 \
    --input_nc 1 \
    --batch_size 1
```

The script runs **A → B and B → A**, prints the mean SSIM for each direction, and saves generated images with HTML galleries:

```text
results/IXI_CRAVIM/test_info/
├── test_100_A2B/
│   ├── index.html
│   └── images/
└── test_100_B2A/
    ├── index.html
    └── images/
```

Keep `--batch_size 1` for the current test implementation. Pretrained checkpoints are not included; train the models before testing. The released test script computes SSIM; the paper reports additional metrics and its full experimental protocol.

## Citation

If this work is useful for your research, please cite the paper. The entry below follows the [MICCAI 2025 open-access citation](https://papers.miccai.org/miccai-2025/0425-Paper1652.html).

```bibtex
@inproceedings{zhang2025cravim,
  title     = {Hybrid State-Space Models and Denoising Training for Unpaired Medical Image Synthesis},
  author    = {Zhang, Junming and Jiang, Shancheng},
  booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2025},
  series    = {Lecture Notes in Computer Science},
  volume    = {15961},
  pages     = {239--248},
  year      = {2025},
  publisher = {Springer Nature Switzerland},
  doi       = {10.1007/978-3-032-04937-7_23}
}
```

## License and acknowledgments

See [LICENSE](LICENSE) and the notices in individual source files for licensing information. Please retain the applicable third-party notices when reusing code.

We thank the authors of [Mamba](https://github.com/state-spaces/mamba), [EfficientVMamba](https://github.com/TerryPei/EfficientVMamba), and [Reg-GAN](https://github.com/Kid-Liet/Reg-GAN) for their open-source work.

For questions or problems with the code, please open a [GitHub issue](https://github.com/jmzhang-cv/CRAViM/issues).
