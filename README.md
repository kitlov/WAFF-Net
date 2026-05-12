# WAFF-Net: Wavelet-Based Adaptive Feature Fusion Lightweight Network for Single Image Dehazing
## Abstract

Image dehazing is a fundamental task in computer vision, yet existing dehazing models commonly face a trade-off between performance and computational complexity. Lightweight frequency-domain dehazing models suffer from imbalanced generalization across synthetic and real-world haze scenes and insufficient detail recovery, failing to meet real-time dehazing demands on resource-constrained devices. To address these issues, we propose WAFF-Net, Wavelet-Based Adaptive Feature Fusion Lightweight Network for Single Image Dehazing that balances dehazing efficacy and inference efficiency. WAFF-Net leverages discrete wavelet decomposition to decouple low- and high-frequency features,introduces a wavelet attention module for noise suppression and lossless downsampling,and adopts a dynamic weight fusion strategy to enhance feature representation and detail recovery. It also constructs a linear-complexity global reconstruction path by fusing wavelet transforms with CNNs, effectively mitigating the limitations of conventional frequency-domain models on synthetic haze scenes. Extensive experiments on real-world (NH-Haze, O-Haze, Dense-Haze) and synthetic (HSTS) haze datasets demonstrate that WAFF-Net achieves superior cross-scene generalization and detail recovery, with its PSNR and SSIM surpassing state-of-the-art lightweight dehazing models while only having 0.28M parameters. This makes WAFF-Net an efficient solution for real-time dehazing on resource-constrained edge devices.

### Requirements

- **CUDA**: 12.8
- **Python**: 3.9.21
- **PyTorch**: 2.0.0

- ### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/WAFF-Net.git
cd WAFF-Net

# Create conda environment
conda create -n waffnet python=3.9.21
conda activate waffnet

# Install PyTorch with CUDA 12.8 support
pip install torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt


