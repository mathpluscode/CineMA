<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="logo_dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="logo_light.svg">
    <img alt="CineMA logo" src="logo_light.svg" height="256">
  </picture>

![python](https://img.shields.io/badge/Python-3.11-3776AB.svg?style=flat&logo=python&logoColor=white)
![pytorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Pre-commit](https://github.com/mathpluscode/CineMA/actions/workflows/pre-commit.yml/badge.svg)
![Pytest](https://github.com/mathpluscode/CineMA/actions/workflows/pytest.yml/badge.svg)
[![codecov](https://codecov.io/gh/mathpluscode/CineMA/graph/badge.svg?token=MZVAOAWUPV)](https://codecov.io/gh/mathpluscode/CineMA)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

# CineMA: A Foundation Model for Cine Cardiac MRI 🎥🫀

> [!TIP]
>
> Check out our [interactive demos](https://huggingface.co/spaces/mathpluscode/CineMA) on Hugging Face to see CineMA in
> action!
>
> Multiple training and inference examples/scripts have also been provided in
> [cinema/examples](https://github.com/mathpluscode/CineMA/tree/main/cinema/examples).

## 📝 Overview

**CineMA** is a vision foundation model for **Cine** cardiac magnetic resonance (CMR) imaging, built on
**M**asked-**A**utoencoder. Pre-trained on the extensive UK Biobank dataset, CineMA has been fine-tuned for various
clinically relevant tasks:

- 🫀 Ventricle and myocardium segmentation
- 📊 Ejection fraction (EF) regression
- 🏥 Cardiovascular disease (CVD) detection and classification
- 📍 Mid-valve plane and apical landmark localization

The model has demonstrated improved or comparative performance against convolutional neural network baselines (UNet,
ResNet) across multiple datasets, including [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/),
[M&Ms](https://www.ub.edu/mnms/), [M&Ms2](https://www.ub.edu/mnms-2/),
[Kaggle](https://www.kaggle.com/c/second-annual-data-science-bowl/data),
[Rescan](https://www.ahajournals.org/doi/full/10.1161/CIRCIMAGING.119.009214), and
[Landmark](https://pubs.rsna.org/doi/10.1148/ryai.2021200197).

## 🚀 Getting Started

### Installation

#### Option 1: Quick Install with pip

```bash
pip install git+https://github.com/mathpluscode/CineMA
```

> Note: This method does not install dependencies automatically.

#### Option 2: Full Installation with Dependencies

```bash
git clone https://github.com/mathpluscode/CineMA.git
cd CineMA
conda env update -f envs/environment.yml
conda activate cinema
pip install -e .
```

> ⚠️ **Important**: Install [PyTorch](https://pytorch.org/get-started/locally/) separately following the official
> instructions.

### 🎯 Using Fine-tuned Models

All fine-tuned models are available on [HuggingFace](https://huggingface.co/mathpluscode/CineMA). Try them out using our
example inference scripts:

```bash
# Segmentation
python examples/inference/segmentation_sax.py
python examples/inference/segmentation_lax_4c.py

# Classification
python examples/inference/classification_cvd.py
python examples/inference/classification_sex.py
python examples/inference/classification_vendor.py

# Regression
python examples/inference/regression_ef.py
python examples/inference/regression_bmi.py
python examples/inference/regression_age.py

# Landmark Detection
python examples/inference/landmark_heatmap.py
python examples/inference/landmark_coordinate.py
```

Available tasks and models are listed below.

| Task                                            | Input View       | Input Timeframes | Inference Script                                                               |
| ----------------------------------------------- | ---------------- | ---------------- | ------------------------------------------------------------------------------ |
| Ventricle and myocardium segmentation           | SAX              | 1                | [segmentation_sax.py](cinema/examples/inference/segmentation_sax.py)           |
| Ventricle and myocardium segmentation           | LAX 4C           | 1                | [segmentation_lax_4c.py](cinema/examples/inference/segmentation_lax_4c.py)     |
| CVD classification                              | SAX or LAX 4C    | 2 (ED and ES)    | [classification_cvd.py](cinema/examples/inference/classification_cvd.py)       |
| Patient sex classification                      | SAX              | 2 (ED and ES)    | [classification_sex.py](cinema/examples/inference/classification_sex.py)       |
| CMR machine vendor classification               | SAX or LAX 4C    | 2 (ED and ES)    | [classification_vendor.py](cinema/examples/inference/classification_vendor.py) |
| EF regression                                   | SAX or LAX 4C    | 2 (ED and ES)    | [regression_ef.py](cinema/examples/inference/regression_ef.py)                 |
| Patient BMI regression                          | SAX              | 2 (ED and ES)    | [regression_bmi.py](cinema/examples/inference/regression_bmi.py)               |
| Patient age regression                          | SAX              | 2 (ED and ES)    | [regression_age.py](cinema/examples/inference/regression_age.py)               |
| Landmark localization by heatmap regression     | LAX 2C or LAX 4C | 1                | [landmark_heatmap.py](cinema/examples/inference/landmark_heatmap.py)           |
| Landmark localization by coordinates regression | LAX 2C or LAX 4C | 1                | [landmark_coordinate.py](cinema/examples/inference/landmark_coordinate.py)     |

### 🔧 Using Pre-trained Models

The pre-trained CineMA backbone is available at [HuggingFace](https://huggingface.co/mathpluscode/CineMA). Fine-tune it
using our example scripts and the [preprocessed ACDC dataset](https://huggingface.co/datasets/mathpluscode/ACDC) for
following tasks:

| Task                                  | Fine-tuning Script                                           |
| ------------------------------------- | ------------------------------------------------------------ |
| Ventricle and myocardium segmentation | [segmentation.py](cinema/examples/train/segmentation.py)     |
| Cardiovascular disease classification | [classification.py](cinema/examples/train/classification.py) |
| Ejection fraction regression          | [regression.py](cinema/examples/train/regression.py)         |

The commandlines are:

```bash
# Fine-tuning Scripts
python examples/train/segmentation.py
python examples/train/classification.py
python examples/train/regression.py
```

You can also explore the reconstruction performance of extract features using following example scripts.

```bash
# MAE Examples
python examples/inference/mae.py
python examples/inference/mae_feature_extraction.py
```

### 📚 Dataset Support

For fine-tuning CineMA on other datasets, pre-process can be performed using the provided scripts following the
documentations. It is recommended to download the data under `~/.cache/cinema_datasets` as the integration tests uses
this path. For instance, the mnms preprocessed data would be `~/.cache/cinema_datasets/mnms/processed`. Otherwise define
the path using environment variable `CINEMA_DATA_DIR`.

| Dataset   | Documentation                                |
| --------- | -------------------------------------------- |
| ACDC      | [README.md](cinema/data/acdc/README.md)      |
| M&Ms      | [README.md](cinema/data/mnms/README.md)      |
| M&Ms2     | [README.md](cinema/data/mnms2/README.md)     |
| Kaggle    | [README.md](cinema/data/kaggle/README.md)    |
| Rescan    | [README.md](cinema/data/rescan/README.md)    |
| Landmark  | [README.md](cinema/data/landmark/README.md)  |
| EMIDEC    | [README.md](cinema/data/emidec/README.md)    |
| Myops2020 | [README.md](cinema/data/myops2020/README.md) |

The code for training and evaluating models on these datasets are available.

| Task                                            | Data      | Documentation                                                                      |
| ----------------------------------------------- | --------- | ---------------------------------------------------------------------------------- |
| Ventricle and myocardium segmentation           | ACDC      | [cinema/segmentation/acdc/README.md](cinema/segmentation/acdc/README.md)           |
| Ventricle and myocardium segmentation           | M&Ms      | [cinema/segmentation/mnms/README.md](cinema/segmentation/mnms/README.md)           |
| Ventricle and myocardium segmentation           | M&Ms2     | [cinema/segmentation/mnms2/README.md](cinema/segmentation/mnms2/README.md)         |
| Ventricle and myocardium segmentation           | Kaggle    | [cinema/segmentation/kaggle/README.md](cinema/segmentation/kaggle/README.md)       |
| Ventricle and myocardium segmentation           | Rescan    | [cinema/segmentation/rescan/README.md](cinema/segmentation/rescan/README.md)       |
| Scar segmentation                               | EMIDEC    | [cinema/segmentation/emidec/README.md](cinema/segmentation/emidec/README.md)       |
| Scar segmentation                               | Myops2020 | [cinema/segmentation/myops2020/README.md](cinema/segmentation/myops2020/README.md) |
| Image classification                            | ACDC      | [cinema/classification/acdc/README.md](cinema/classification/acdc/README.md)       |
| Image classification                            | M&Ms      | [cinema/classification/mnms/README.md](cinema/classification/mnms/README.md)       |
| Image classification                            | M&Ms2     | [cinema/classification/mnms2/README.md](cinema/classification/mnms2/README.md)     |
| Image regression                                | ACDC      | [cinema/regression/acdc/README.md](cinema/regression/acdc/README.md)               |
| Image regression                                | M&Ms      | [cinema/regression/mnms/README.md](cinema/regression/mnms/README.md)               |
| Image regression                                | M&Ms2     | [cinema/regression/mnms2/README.md](cinema/regression/mnms2/README.md)             |
| Landmark localization by heatmap regression     | Landmark  | [cinema/segmentation/landmark/README.md](cinema/segmentation/landmark/README.md)   |
| Landmark localization by coordinates regression | Landmark  | [cinema/regression/landmark/README.md](cinema/regression/landmark/README.md)       |

### 🏗️ Training Your Own Foundation Model

Start with our simplified pretraining script:

```bash
python examples/train/pretrain.py
```

For distributed training support, check [cinema/mae/pretrain.py](cinema/mae/pretrain.py).

## 📖 References

CineMA builds upon these open-source projects:

- [UK Biobank Cardiac Preprocessing](https://github.com/baiwenjia/ukbb_cardiac)
- [Masked Autoencoders](https://github.com/facebookresearch/mae)
- [ConvMAE](https://github.com/Alpha-VL/ConvMAE)
- [MultiMAE](https://github.com/EPFL-VILAB/MultiMAE)
- [MONAI](https://github.com/Project-MONAI/MONAI)
- [PyTorch Vision](https://github.com/pytorch/vision)
- [PyTorch Image Models](https://github.com/huggingface/pytorch-image-models)

## 🤝 Contributing

We welcome contributions! Please [create an issue](https://github.com/mathpluscode/CineMA/issues/new) for questions or
suggestions.

## 📧 Contact

For collaborations, reach out to Yunguan Fu (yunguan.fu.18@ucl.ac.uk).

## 📄 Citation

```
@article{fu2025cinema,
  title={CineMA: A Foundation Model for Cine Cardiac MRI},
  author={Fu, Yunguan and Yi, Weixi and Manisty, Charlotte and Bhuva, Anish N and Treibel, Thomas A and Moon, James C and Clarkson, Matthew J and Davies, Rhodri Huw and Hu, Yipeng},
  journal={arXiv preprint arXiv:2506.00679},
  year={2025}
}
```
