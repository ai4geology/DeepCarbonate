
<!-- 顶部徽章区域 -->
<div align="center">

##  DeepCarbonate: A Dataset and Benchmark of Carbonate Thin-Section Images for Deep Learning
</div>

Keran Li<sup>a,1,2</sup>, Jinmin Song<sup>a,1</sup>, Zhaoyi Zhang<sup>a,1</sup>, Shan Ren<sup>a,1</sup>, Yang Lan<sup>b,1</sup>, Di Yang<sup>a</sup>, Zhiwu Li<sup>a</sup>, Shugen Liu<sup>a</sup>, Chunqiao Yan<sup>a</sup>, Xin Jin<sup>a</sup>, Shaohai Yang<sup>a</sup>, Jiaxin Guo<sup>a</sup>

<sup>a</sup>State Key Laboratory of Oil and Gas Reservoir Geology and Exploitation, Chengdu University of 
Technology, Chengdu 610059, China

<sup>b</sup>School of Economics and Management, Beihang University, Beijing, 100191, China


<sup>1</sup>Equal contribution

<sup>2</sup>Present address: State Key Laboratory of Critical Earth Material Cycling and Mineral Deposits, Frontiers Science Center for Critical Earth Material Cycling, School of Earth Sciences and Engineering, Nanjing University, Nanjing, 210023, China

<sup>*</sup>Corresponding authors

---

[![Scientific Data](https://img.shields.io/badge/Nature%20Sci.%20Data-Published-1abc9c.svg?style=flat-square&logo=nature&logoColor=white)](https://www.nature.com/articles/s41597-026-06633-5)
[![Zenodo Dataset](https://img.shields.io/badge/Zenodo-55,786%20Images-blue.svg?style=flat-square&logo=zenodo&logoColor=white)](https://doi.org/10.5281/zenodo.18061204)
[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg?style=flat-square)](https://creativecommons.org/licenses/by-nc-nd/4.0/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)

**33.2 GB · 55,786 Images · 22 Lithological Categories · 3 Optical Modes**

[📄 Paper](https://www.nature.com/articles/s41597-026-06633-5) • 
[💾 Dataset (Zenodo)](https://doi.org/10.5281/zenodo.18061204) • 
[🌐 Project Page](https://github.com/KeranLi/DeepCarbonate) • 
[📊 Benchmarks](#benchmark-results)


---

## 📋 Table of Contents

- [Overview](#-overview)
- [Geological Background](#-geological-background)
- [Dataset Statistics](#-dataset-statistics)
- [Installation](#️-installation)
- [Usage](#-usage)
- [Benchmark Results](#-benchmark-results)
- [Citation](#-citation)

---

## 🎯 Overview

DeepCarbonate is a rigorously curated dataset for **automated carbonate petrography** using deep learning. Unlike generic image datasets, it addresses the multimodal nature of geological thin-section analysis requiring specific optical conditions (plane-polarized, cross-polarized, and reflected light) to distinguish mineralogical features.

### 🔬 Key Features
- **Expert-validated**: 10 petroleum geologists (6 Ph.D., 4 M.Sc.) voted on lithological labels; images with >20% disagreement excluded
- **Multimodal imaging**: PPL, XPL, and Reflected light modes for comprehensive mineralogical characterization
- **Standardized format**: Organized in ImageNet-style hierarchy (`train/val/test`)
- **Quality controlled**: 345 invalid images removed via 2σ filtering (blur, brightness, artifacts) + expert inspection
- **Geologically diverse**: Covers microbialites, grainstones, evaporites, and reservoir features (pores, fractures)

---

## 🌍 Geological Background

Samples were collected from carbonate reservoirs spanning **~500 million years** of geological history:

| Basin | Formation | Age | Lithology Focus |
|:-----:|:---------:|:---:|:---------------|
| **Sichuan Basin, China** | Dengying | Ediacaran | Microbial carbonates, dolostones |
| **Sichuan Basin, China** | Longwangmiao | Cambrian | Grainstones, oolites |
| **Sichuan Basin, China** | Leikoupo & Jialingjiang | Triassic | Evaporites, lagoonal deposits |
| **UAE** | Mishrif | Cretaceous | Reefal limestones, bioclastic carbonates |
| **Others** |Unkown | Unkown | Reefal limestones, bioclastic carbonates |

### 🏔️ 22 Lithological Categories

<details>
<summary>Click to expand class definitions (Folk & Dunham classification)</summary>

| Class ID | Category | Description | Typical Features |
|:--------:|:---------|:------------|:----------------|
| 1 | Microbial mudstone | Gelatinous EPS-rich matrix | Micrite, microbial textures |
| 2 | Bioclastic carbonates | Shell/coral fragments | Allochem-rich, variable sorting |
| 3 | Breccia carbonates | Angular >2mm fragments | Tectonic/collapse features |
| 4 | Coarse crystal dolostone | Recrystallized dolomite | Euhedral crystals >100μm |
| 5 | Coarse crystal limestone | Sparitic calcite | Diagenetic recrystallization |
| 6 | Evaporite | Halite/gypsum precipitation | Arid environment indicator |
| 7 | Finely crystal dolostone | Early diagenetic dolomite | <10μm crystals, anhydrite assoc. |
| 8 | Finely crystal limestone | Microcrystalline calcite | Hypersaline pore waters |
| 9 | Foam carbonates | 3D interconnected pore network | Microbial, high porosity |
| 10 | Cemented fracture | Calcite/silica filled fractures | Reservoir connectivity features |
| 11 | Botryoidal dolostone | Grape-like textures | Microbial precipitates |
| 12 | Micritic dolostone | <4μm dolomite crystals | Restricted marine environments |
| 13 | Micritic limestone | <63μm calcite matrix | Low permeability seal rocks |
| 14 | Mid crystal dolostone | Intermediate crystallinity | 10-100μm crystals |
| 15 | Mid crystal limestone | Diagenetic texture | Moderate porosity |
| 16 | Oncolite | Spherical concentric structures | Nucleus-coated grains |
| 17 | Oolite | <2mm concentric ooids | High-energy shallow marine |
| 18 | Siliceous carbonates | Chert/opal bearing | Silica diagenesis |
| 19 | Stromatolites | Laminated microbial mats | Layered cyanobacterial structures |
| 20 | Stylolite | Pressure solution seams | Irregular interlocking columns |
| 21 | Thrombolite | Clotted microbial structure | Irregular mesoscopic fabric |
| 22 | Pore | Primary/cemented voids | Reservoir quality indicator |

</details>

---

## 📊 Dataset Statistics


### Overall Distribution
| Subset | Images | Percentage | Size |
|:------:|:------:|:----------:|:----:|
| **Train** | 39,070 | 70.0% | ~23.2 GB |
| **Validation** | 11,157 | 20.0% | ~6.6 GB |
| **Test** | 5,559 | 10.0% | ~3.4 GB |
| **Total** | **55,786** | 100% | **33.2 GB** |


### Detailed Class Statistics

| Rank | Class | Total Images | Train | Val | Test | Optical Modes |
|:----:|:------|:------------:|:-----:|:---:|:----:|:-------------:|
| 1 | **Pore** | 6,432 | 4,502 | 1,286 | 644 | PPL, XPL |
| 2 | **Thrombolite** | 5,328 | 3,730 | 1,066 | 532 | PPL, XPL |
| 3 | **Microbial mudstone** | 4,728 | 3,310 | 946 | 472 | PPL, XPL |
| 4 | **Stromatolites** | 3,569 | 2,498 | 714 | 357 | PPL, XPL |
| 5 | **Micritic limestone** | 3,231 | 2,262 | 646 | 323 | PPL, XPL, ARS |
| 6 | **Bioclastic carbonates** | 3,031 | 2,122 | 606 | 303 | PPL, XPL |
| 7 | **Micritic dolostone** | 2,974 | 2,082 | 595 | 297 | PPL, XPL, ARS |
| 8 | **Siliceous carbonates** | 2,938 | 2,057 | 588 | 293 | PPL, XPL, R |
| 9 | **Cemented fracture** | 2,788 | 1,952 | 558 | 278 | PPL, XPL, R |
| 10 | Foam carbonates | 1,218 | 853 | 244 | 121 | PPL, XPL |
| 11 | Breccia carbonates | 1,123 | 786 | 225 | 112 | PPL, XPL |
| 12 | Stylolite | 917 | 642 | 183 | 92 | PPL, XPL |
| 13 | Evaporite | 575 | 403 | 115 | 57 | PPL, XPL |
| 14 | Mid crystal dolostone | 424 | 297 | 85 | 42 | PPL, XPL, ARS |
| 15 | Botryoidal dolostone | 252 | 176 | 50 | 26 | PPL, XPL |
| 16 | Finely crystal dolostone | 234 | 164 | 47 | 23 | PPL, XPL |
| 17 | Oolite | 150 | 105 | 30 | 15 | PPL, XPL |
| 18 | Oncolite | 129 | 90 | 26 | 13 | PPL, XPL |
| 19 | Coarse crystal dolostone | 112 | 78 | 22 | 12 | PPL, XPL |
| 20 | Coarse crystal limestone | 108 | 76 | 22 | 10 | PPL, XPL |
| 21 | Mid crystal limestone | 14 | 10 | 3 | 1 | PPL, XPL |
| 22 | Finely crystal limestone | 14 | 10 | 3 | 1 | PPL, XPL |

> **Note**: PPL = Plane-Polarized Light, XPL = Cross-Polarized Light, R = Reflected Light, ARS = Alizarin Red Stained

---

## ⚙️ Installation

### Prerequisites
- **Hardware**: NVIDIA GPU with ≥80GB VRAM (tested on A800) for full training; 16GB+ for inference
- **Software**: Python 3.8+, CUDA 11.3+

### Environment Setup

```bash
# Clone repository
git clone https://github.com/KeranLi/DeepCarbonate.git
cd DeepCarbonate

# Create conda environment
conda create -n deepcarbonate python=3.9
conda activate deepcarbonate

# Install core dependencies
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

**Key Dependencies:**
```text
torch >= 1.11.0
torchvision >= 0.12.0
numpy >= 1.20.0
pillow >= 8.0.0
matplotlib >= 3.3.0
scikit-learn >= 0.24.0
tqdm >= 4.60.0
pandas >= 1.3.0
seaborn >= 0.11.0
```

---

## 🚀 Usage

### 1. Data Preparation

Download the dataset from [Zenodo](https://doi.org/10.5281/zenodo.18061204) and organize as follows:

```text
DeepCarbonate/
├── PPL/                    # Plane-Polarized Light (~18,595 images)
│   ├── train/
│   │   ├── class1/         # Microbial mudstone
│   │   ├── class2/         # Bioclastic carbonates
│   │   └── ...             # class3-22
│   ├── val/
│   └── test/
├── XPL/                    # Cross-Polarized Light (~18,595 images)
│   ├── train/
│   ├── val/
│   └── test/
└── R/                      # Reflected Light (~4,000 images)
    ├── train/
    ├── val/
    └── test/
```

**Naming Convention**: `LithologyName.Index.jpg` (e.g., `Bioclastic carbonates.2.jpg`, `Pore.22.jpg`)

### 2. Training

#### Full 22-Class Training (Baseline)
```bash
python main.py \
    --mode train \
    --model resnet50 \
    --data_path ./PPL \
    --epochs 50 \
    --batch_size 512 \
    --lr 1e-5 \
    --num_classes 22 \
    --early_stopping
```

#### Top-9 Balanced Subset (Recommended for Stable Training)
```bash
python main.py \
    --mode train \
    --model resnet50 \
    --data_path ./PPL \
    --classes 1,2,3,4,5,6,7,8,9 \
    --epochs 50 \
    --batch_size 256
```

#### Ablation: Single Polarization Mode
```bash
# Train with PPL only (expect performance drop)
python main.py --mode train --model mobilenet --data_path ./PPL --ablation_mode single_polar

# Train without Alizarin Red Stained (ARS) images
python main.py --mode train --model resnet18 --exclude_ars
```

### 3. Evaluation

```bash
# Generate confusion matrix and full metrics
python eval.py \
    --checkpoint experiments/best_model.pth \
    --data_path ./PPL/test \
    --num_classes 22 \
    --save_results ./results/
```

### 4. Class Activation Mapping (CAM) Visualization

```bash
python visualize_cam.py \
    --image path/to/image.jpg \
    --model resnet50 \
    --checkpoint experiments/best_model.pth
```

---

## 🏆 Benchmark Results

### Main Results: 22-Class Classification


| Model | Accuracy | Precision | Recall | F1-Score | AUC | Top-1 Acc | Top-5 Acc |
|:-----:|:--------:|:---------:|:------:|:--------:|:---:|:---------:|:---------:|
| **MobileNet** | **0.68** | **0.68** | **0.67** | **0.67** | **0.67** | 0.52 | 0.81 |
| **DenseNet** | 0.64 | 0.64 | 0.63 | 0.64 | 0.64 | 0.55 | 0.82 |
| **VGG16** | 0.62 | 0.62 | 0.61 | 0.62 | 0.63 | 0.57 | 0.84 |
| **EfficientNet** | 0.63 | 0.62 | 0.62 | 0.62 | 0.62 | 0.52 | 0.81 |
| **ResNet18** | 0.62 | 0.62 | 0.63 | 0.63 | 0.62 | 0.56 | 0.81 |
| ResNet50 | 0.56 | 0.57 | 0.55 | 0.57 | 0.56 | **0.58** | **0.85** |


**Key Findings:**
- **MobileNet** achieves best overall metrics (Acc/F1: 0.68) despite being lightweight
- **ResNet50** shows highest Top-5 accuracy (85%) but lowest Top-1 (56%), indicating hierarchical confusion
- **Long-tail effect**: Severe performance degradation on tail classes (14-112 samples) vs head classes (6,000+ samples)

### Top-9 Balanced Subset Results

| Model | Accuracy | Precision | Recall | F1-Score | AUC | Top-1 Acc | Top-9 Acc |
|:-----:|:--------:|:---------:|:------:|:--------:|:---:|:---------:|:---------:|
| **ResNet50** | **0.85** | **0.85** | **0.85** | **0.86** | **0.85** | 0.72 | 0.90 |
| ResNet18 | 0.83 | 0.82 | 0.83 | 0.83 | 0.83 | **0.78** | 0.81 |
| EfficientNet | 0.75 | 0.72 | 0.72 | 0.73 | 0.74 | 0.74 | 0.84 |
| DenseNet | 0.73 | 0.73 | 0.73 | 0.74 | 0.74 | 0.74 | 0.86 |
| MobileNet | 0.72 | 0.72 | 0.71 | 0.70 | 0.72 | 0.72 | 0.85 |
| VGG16 | 0.72 | 0.82 | 0.72 | 0.71 | 0.72 | 0.67 | 0.88 |

### Ablation Studies

| Configuration | MobileNet Acc | ResNet50 Acc | Finding |
|:-------------|:-------------:|:------------:|:--------|
| **All Modes** | 0.68 | 0.56 | Baseline (multimodal) |
| **w/o ARS** | 0.70 | 0.58 | Alizarin Red staining removal has minimal impact |
| **PPL Only** | 0.51 | 0.49 | Single polarization causes severe feature loss |
| **XPL Only** | 0.52 | 0.50 | Cross-polarized alone insufficient |
| **R Only** | 0.48 | 0.46 | Reflected light limited for mineral ID |

---

## 🖼️ Data Quality Control

### Image Quality Metrics (2σ Filtering)
Following strict geological imaging standards:

| Metric | Threshold | Rejected |
|:-------|:---------:|:--------:|
| Brightness | μ ± 2σ | 152 images (low brightness) |
| Sharpness | Laplacian variance | 174 images (blur) |
| Blur | Gaussian detection | Motion artifacts |
| Artifacts | Noise detection | 19 images (damaged) |

**Final Clean Dataset**: 55,441 valid images (99.4% retention rate)

### Microscopy Parameters
- **Equipment**: Nikon Eclipse E600POL
- **Illumination**: Köhler method, 12V 100W halogen, stabilized at 9.8V
- **CCD**: 14-bit, 0.63× C-mount
- **Magnification**: 10× (primary), 5× (overview)
- **Optics**: Field/aperture diaphragms at 90%/70% NA (transmitted), 60% (reflected)

---

## 📚 Citation

If you use DeepCarbonate in your research, please cite:

```bibtex
@article{li2026deepcarbonate,
  title={A dataset and benchmark of carbonate thin-section images for deep learning},
  author={Li, Keran and Song, Jinmin and Zhang, Zhaoyi and Ren, Shan and Lan, Yang and Yang, Di and Li, Zhiwu and Liu, Shugen and Yan, Chunqiao and Jin, Xin and Yang, Shaohai and Guo, Jiaxin},
  journal={Scientific Data},
  volume={X},
  pages={XXX--XXX},
  year={2026},
  publisher={Nature Publishing Group},
  doi={10.1038/s41597-026-06633-5}
}
```

**Dataset Citation**:
```bibtex
@dataset{li2025deepcarbonate,
  author={Li, Keran and Song, Jinmin and Zhang, Zhaoyi and Ren, Shan and Lan, Yang and Yang, Di and Li, Zhiwu and Liu, Shugen and Yan, Chunqiao and Jin, Xin and Yang, Shaohai and Guo, Jiaxin},
  title={DeepCarbonate: A benchmark dataset for carbonate thin-section image analysis},
  year={2025},
  publisher={Zenodo},
  doi={10.5281/zenodo.18061204}
}
```

---

## ⚠️ Usage Notes

1. **Long-tail Distribution**: The dataset exhibits severe class imbalance (6432 vs 14 images). For production use, we strongly recommend:
   - Using the **Top-9 balanced subset** for initial model development
   - Applying class-weighted loss functions or oversampling for full 22-class training

2. **Optimal Imaging**: For inference on new samples, capture both **PPL and XPL** images simultaneously. Reflected light (R) is supplementary for organic matter/bitumen analysis.

3. **Magnification**: All images captured at **10× magnification** under standardized Köhler illumination.

4. **Staining**: Alizarin Red S staining available for subset; use for distinguishing calcite vs dolomite.

---

## 📂 Project Structure

```text
DeepCarbonate/
├── 📂 data/                       # Dataset directory (download separately)
│   ├── PPL/                       # Plane-polarized light images
│   ├── XPL/                       # Cross-polarized light images
│   └── R/                         # Reflected light images
├── 📂 experiments/                # Training outputs
│   ├── models/                    # Saved checkpoints (.pth)
│   ├── logs/                      # TensorBoard logs
│   └── results/                   # Evaluation metrics
├── 📂 src/                        # Source code
│   ├── dataset.py                 # Custom Dataset class
│   ├── model.py                   # ResNet, DenseNet, EfficientNet, MobileNet
│   ├── trainer.py                 # Training loop with early stopping
│   └── eval.py                    # Metrics calculation
├── main.py                        # Entry point
├── requirements.txt               # Dependencies
├── LICENSE                        # CC BY-NC-ND 4.0 for data, MIT for code
└── README.md                      # This file
```

---

## 🤝 Contributing

We welcome contributions to expand the dataset, especially:
- Additional samples for underrepresented classes (Classes 14-22)
- New optical modes (fluorescence, cathodoluminescence)
- Segmentation masks for mineral phase analysis

## 📝 License

- **Dataset**: [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) (Non-commercial use only)
- **Code**: MIT License