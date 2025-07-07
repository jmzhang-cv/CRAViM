# CRAViM

This repository contains the code for the MICCAI 2025 paper: "Hybrid State-Space Models and Denoising Training for Unpaired Medical Image Synthesis."

The source code will be uploaded to this GitHub repository at a later date.

---

## Environment Setup

This project requires **Python 3.10.8** and **PyTorch 2.1.2 (with CUDA 12.1 support)**.

**Please note:** The configuration of the `mamba` environment can be challenging. We advise users to proceed with caution. Below are the specific environment details currently employed for this project:

    torch==2.1.2+cu121

    causal_conv1d @ https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.0.post8/causal_conv1d-1.5.0.post8%2Bcu12torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

    mamba_ssm @ https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4%2Bcu12torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

## Datasets
Due to patient privacy concerns, we are unable to publicly release the datasets used in this study. However, we recommend utilizing the pre-processed data provided in the following repository for your research:

    https://github.com/Kid-Liet/Reg-GAN

Please ensure you obtain the necessary license from the BraTS dataset website before using the data.
