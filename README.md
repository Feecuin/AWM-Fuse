# AWM-Fuse: Unified Model for Infrared-Visible Fusion and Compound Adverse-Weather Restoration

[![arXiv](https://img.shields.io/badge/arXiv-2603.02560-b31b1b.svg)](https://arxiv.org/abs/2603.02560)
[![Python](https://img.shields.io/badge/Python-3.8%2B-yellow.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red.svg)](https://pytorch.org/)

> **Authors: Xilai Li, Huichun Liu(Co-first Authors), Xiaosong Li, Tao Ye, Zhenyu Kuang, Huafeng Li**

This is the official PyTorch implementation of **AWM-Fuse**, a unified framework that handles diverse weather degradations via global and local text perception with shared parameters. AWM-Fuse integrates three key components: (1) **Global Text Perception Module (GTPM)** to extract scene features and degradation types; (2) **Local Text Perception Module (LTPM)** to concentrate on specific degradation effects; and (3) **VLM-Driven Loss** to enforce semantic consistency between fused outputs and textual descriptions.

<p align="center">
  <img src="Figs/Fig1.pdf" width="90%">
</p>

## 📖 Abstract

Multi-modality image fusion (MMIF) in adverse weather aims to address the loss of visual information caused by weather-related degradations, providing clearer scene representations. Although a few studies have attempted to incorporate textual information to improve semantic perception, they often lack effective categorization and thorough analysis of textual content. To address these limitations, we propose **AWM-Fuse**, a unified fusion framework that handles diverse weather degradations via global and local text perception with shared parameters. In particular, a global feature perception module leverages BLIPgenerated captions to extract overall scene features and identify primary degradation types, thus promoting generalization across various adverse weather conditions. Complementing this, the local module employs detailed scene descriptions produced by ChatGPT to concentrate on specific degradation effects through concrete textual cues, enabling the recovery of subtle details. Furthermore, textual descriptions are used to constrain the generation of fusion images, effectively steering the network learning process toward better alignment with semantic labels, thereby promoting the learning of more meaningful visual features. To facilitate text-guided fusion under adverse weather, we construct **AWMM-Text**, a large-scale benchmark providing paired global and local annotations for multi-modality image pairs. Extensive experiments demonstrate that AWM-Fuse consistently outperforms state-of-the-art methods under complex weather conditions and on multiple downstream tasks.

## ✨ Highlights

- 🎯 **Unified Text-Guided Framework**: Joint global and local text perception for diverse weather degradations with shared parameters
- 📝 **Global Text Perception Module**: Leverages BLIP-generated captions to extract scene structure and primary degradation types
- 🔤 **Local Text Perception Module**: Employs LLM-generated detailed descriptions for precise local degradation guidance and detail recovery
- 🌊 **VLM-Driven Loss**: CLIP-based image–text alignment ensures semantic consistency between fusion outputs and textual descriptions
- 🎯 **Semantic-Constrained Learning**: Text constraints steer network training toward better semantic label alignment
- 📊 **AWMM-Text Benchmark**: Large-scale paired global/local textual annotations for adverse-weather multi-modality images
- ⚡ **State-of-the-Art Performance**: Consistently outperforms SOTA methods on diverse weather conditions and downstream tasks
- 🔄 **Robust Cross-Modal Interaction**: Effective fusion despite severe visible image degradation and feature distribution changes

## 🔥 Visual Results

### Rain Scenarios

<p align="center">
  <img src="Figs/Fig3.pdf" width="90%">
</p>

### Haze Scenarios

<p align="center">
  <img src="Figs/Fig5.pdf" width="90%">
</p>

### Snow Scenarios

<p align="center">
  <img src="Figs/Fig4.pdf" width="90%">
</p>

### Real World Performance

<p align="center">
  <img src="Figs/Fig9.pdf" width="90%">
</p>

### Downstream Task Performance

<p align="center">
  <img src="Figs/Fig7.pdf" width="90%">
</p>

## ✨ Overview

AWM-Fuse combines: (1) global text encoding (BLIP) to capture high-level scene semantics and degradation categories, (2) local text encoding (LLM/ChatGPT) to provide token-level cues for damaged regions, and (3) a cross-modal fusion engine with multi-scale/wavelet processing to decouple frequency degradations. A VLM-driven loss (CLIP) aligns fused outputs with text semantics.

## 📦 Datasets

### Dataset Download

| Source | Dataset |
|:---:|:---:|
| **☁️ Baidu Cloud** | [📥 Download](https://pan.baidu.com/s/17TUs9KbUg1E1YaJ-utyLTQ?pwd=46j6) <br> `pwd: [46j6]` |
|


### Pre-Trained Models

| Source | Weights |
|:---:|:---:|
| **☁️ Baidu Cloud** | [📥 Download](https://pan.baidu.com/s/1T0C73v3ypZH1z9qwG1jOxw?pwd=sbm3) <br> `pwd: [sbm3]` |
|

> 💡 **Tip**: Pre-trained models are available for quick inference and fine-tuning on your own datasets.



### Dataset Structure

Expected directory structure for training data:

```
datasets/
├── train/
│   ├── ir/              # Infrared images
│   ├── vi/              # Visible light images
│   ├── gt_ir/           # Ground truth IR
│   ├── gt_vi/           # Ground truth VI
│   └── Text/
│       ├── caption/     # BLIP-generated captions
│       ├── vi_npy/      # Pre-extracted VI features
│       └── ir_npy/      # Pre-extracted IR features
└── test/                # Similar structure
```

**AWMM-Text Benchmark**: A large-scale dataset with paired global and local textual annotations for adverse-weather multi-modality image pairs, enabling text-guided fusion under diverse weather conditions.

## ⚙️ Installation

### Requirements
- Python >= 3.8
- PyTorch >= 1.9.0
- CUDA (for GPU acceleration)

### Dependencies

```bash
pip install torch torchvision torchaudio
pip install timm einops clip transformers opencv-python
pip install pytorch-wavelets pywt
```

Alternatively, install from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/AWM-Fuse.git
cd AWM-Fuse
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Preprocess (optional)

- Generate BLIP captions and save to `datasets/.../Text/caption/`.
- Optionally pre-extract BLIP/CLIP visual features to `.npy` and place in `Text/vi_npy` and `Text/ir_npy`.

### Training

Edit `Code/train.py` or pass arguments:

```bash
python Code/train.py \
	--ir_path ./datasets/train/ir \
	--vi_path ./datasets/train/vi \
	--gt_path ./datasets/train/gt_vi \
	--gt_ir_path ./datasets/train/gt_ir \
	--clip_path ./datasets/train/Text/caption \
	--blip1_path ./datasets/train/Text/vi_npy \
	--blip2_path ./datasets/train/Text/ir_npy \
	--batchsize 3 --lr 1e-4 --nEpochs 250 --img_size 112
```

### Testing

```bash
python Code/test.py \
	--model_path ./Checkpoint/model_best.pth \
	--ir_path ./datasets/test/ir \
	--vi_path ./datasets/test/vi \
	--clip_path ./datasets/test/Text/caption \
	--save_path ./Experiment/results
```

## 📋 Loss Functions

The project uses a comprehensive multi-component fusion loss function implemented in [Code/losses/losses.py](Code/losses/losses.py):

- **Reconstruction Loss**: L1/L2 distance to preserve pixel-level fidelity
- **Gradient Loss** (`L_Grad`): Maintains edge and structural details from source images
- **SSIM Loss** (`L_SSIM`): Ensures structural similarity with source images
- **Perceptual Loss**: Deep feature-based perceptual quality using pretrained networks (VGG, etc.)
- **CLIP-based VLM Alignment Loss** (`L_CLIP`): Image–text matching to enforce semantic consistency between fused outputs and textual descriptions — key for semantic supervision under severe degradations

The final loss is computed as a weighted combination of all components for optimal multi-modal image fusion under adverse weather conditions:

$$\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{recon} + \lambda_2 \mathcal{L}_{grad} + \lambda_3 \mathcal{L}_{ssim} + \lambda_4 \mathcal{L}_{percep} + \lambda_5 \mathcal{L}_{clip}$$



##  📚Citation

If you use this code and work, please cite:

```bibtex
@article{Li2026AWMFuse,
  title={AWM-Fuse: Multi-Modality Image Fusion for Adverse Weather via Global and Local Text Perception},
  author={Li, Xilai and Liu, Huichun and Li, Xiaosong and Ye, Tao and Kuang, Zhenyu and Li, Huafeng},
  year={2026}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bug reports and feature requests.

## 🙏 Acknowledgments

- Built with [PyTorch](https://pytorch.org/)
- BLIP models from [Salesforce](https://github.com/salesforce/BLIP)
- CLIP models from [OpenAI](https://github.com/openai/CLIP)
- Inspired by recent advances in vision-language models and multi-modal fusion

## 📧 Contact

For questions and inquiries:
- **Email**: Feecuin@163.com
- **GitHub Issues**: [Submit an issue](https://github.com/Feecuin/AWM-Fuse/issues)

---

**Last Updated**: April 2026
