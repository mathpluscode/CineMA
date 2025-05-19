<picture>
  <source media="(prefers-color-scheme: dark)" srcset="logo_dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="logo_light.svg">
  <img alt="CineMA logo" src="logo_light.svg" height="256">
</picture>

# CineMA: A Foundation Model for Cine Cardiac MRI 🎥🫀

![python](https://img.shields.io/badge/Python-3.11-3776AB.svg?style=flat&logo=python&logoColor=white)
![pytorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Pre-commit](https://github.com/mathpluscode/CineMA/actions/workflows/pre-commit.yml/badge.svg)
![Pytest](https://github.com/mathpluscode/CineMA/actions/workflows/pytest.yml/badge.svg)
[![codecov](https://codecov.io/gh/mathpluscode/CineMA/graph/badge.svg?token=MZVAOAWUPV)](https://codecov.io/gh/mathpluscode/CineMA)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

**CineMA** is a vision foundation model for **Cine** cardiac magnetic resonance (CMR) imaging based on
**M**asked-**A**utoencoder. CineMA has been pre-trained on UK Biobank data and fine-tuned on multiple clinically
relevant tasks such as ventricle and myocaridum segmentation, ejection fraction (EF) regression, cardiovascular disease
(CVD) detection and classification, and mid-valve plane and apical landmark localization. The model has been evaluated
on multiple datasets, including [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/),
[M&Ms](https://www.ub.edu/mnms/), [M&Ms2](https://www.ub.edu/mnms-2/),
[Kaggle](https://www.kaggle.com/c/second-annual-data-science-bowl/data),
[Rescan](https://www.ahajournals.org/doi/full/10.1161/CIRCIMAGING.119.009214), and
[Landmark](https://pubs.rsna.org/doi/10.1148/ryai.2021200197), etc.

## Usage

### Installation

You can install the package inside a [Conda](https://github.com/conda-forge/miniforge) environment using following
commands

```bash
git clone https://github.com/mathpluscode/CineMA.git
cd CineMA
conda env update -f envs/environment.yml
conda activate cinema
pip install -e .
```

[Pytorch](https://pytorch.org/get-started/locally/) should be installed separately following the official instructions.

### Use fine-tuned models

The fine-tuned models have been released at https://huggingface.co/mathpluscode/CineMA. Example inference scripts are
available to test these models.

```bash
python examples/inference/segmentation_sax.py
python examples/inference/segmentation_lax_4c.py
python examples/inference/classification_cvd.py
python examples/inference/classification_sex.py
python examples/inference/classification_vendor.py
python examples/inference/regression_ef.py
python examples/inference/regression_bmi.py
python examples/inference/regression_age.py
python examples/inference/landmark_heatmap.py
python examples/inference/landmark_coordinate.py
```

| Training Task                                   | Input View       | Input Timeframes | Inference Script                                                        |
| ----------------------------------------------- | ---------------- | ---------------- | ----------------------------------------------------------------------- |
| Ventricle and myocardium segmentation           | SAX              | 1                | [segmentation_sax.py](examples/inference/segmentation_sax.py)           |
| Ventricle and myocardium segmentation           | LAX 4C           | 1                | [segmentation_lax_4c.py](examples/inference/segmentation_lax_4c.py)     |
| CVD classification                              | SAX or LAX 4C    | 2 (ED and ES)    | [classification_cvd.py](examples/inference/classification_cvd.py)       |
| Patient sex classification                      | SAX              | 2 (ED and ES)    | [classification_sex.py](examples/inference/classification_sex.py)       |
| CMR machine vendor classification               | SAX or LAX 4C    | 2 (ED and ES)    | [classification_vendor.py](examples/inference/classification_vendor.py) |
| EF regression                                   | SAX or LAX 4C    | 2 (ED and ES)    | [regression_ef.py](examples/inference/regression_ef.py)                 |
| Patient BMI regression                          | SAX              | 2 (ED and ES)    | [regression_bmi.py](examples/inference/regression_bmi.py)               |
| Patient age regression                          | SAX              | 2 (ED and ES)    | [regression_age.py](examples/inference/regression_age.py)               |
| Landmark localization by heatmap regression     | LAX 2C or LAX 4C | 1                | [landmark_heatmap.py](examples/inference/landmark_heatmap.py)           |
| Landmark localization by coordinates regression | LAX 2C or LAX 4C | 1                | [landmark_coordinate.py](examples/inference/landmark_coordinate.py)     |

### Use pre-trained models

The pre-trained CineMA model backbone is available at https://huggingface.co/mathpluscode/CineMA. Following scripts
demonstrated how to fine-tune this backbone using
[a preprocessed version of ACDC dataset](https://huggingface.co/datasets/mathpluscode/ACDC):

```bash
python examples/train/segmentation.py
python examples/train/classification.py
python examples/train/regression.py
```

| Task                                  | Fine-tuning Script                                    |
| ------------------------------------- | ----------------------------------------------------- |
| Ventricle and myocardium segmentation | [segmentation.py](examples/train/segmentation.py)     |
| Cardiovascular disease classification | [classification.py](examples/train/classification.py) |
| Ejection fraction regression          | [regression.py](examples/train/regression.py)         |

Another two scripts demonstrated the masking and prediction process of MAE and the feature extraction from MAE.

```bash
python examples/inference/mae.py
python examples/inference/mae_feature_extraction.py
```

For fine-tuning CineMA on other datasets, pre-process can be performed using the provided scripts following the
documentations. Note that it is recommended to download the data under `~/.cache/cinema_datasets` as the integration
tests uses this path. For instance, the mnms preprocessed data would be `~/.cache/cinema_datasets/mnms/processed`.
Otherwise define the path using environment variable `CINEMA_DATA_DIR`.

| Training Data | Documentations                               |
| ------------- | -------------------------------------------- |
| ACDC          | [README.md](cinema/data/acdc/README.md)      |
| M&Ms          | [README.md](cinema/data/mnms/README.md)      |
| M&Ms2         | [README.md](cinema/data/mnms2/README.md)     |
| Kaggle        | [README.md](cinema/data/kaggle/README.md)    |
| Rescan        | [README.md](cinema/data/rescan/README.md)    |
| Landmark      | [README.md](cinema/data/landmark/README.md)  |
| EMIDEC        | [README.md](cinema/data/emidec/README.md)    |
| Myops2020     | [README.md](cinema/data/myops2020/README.md) |

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

### Train your own foundation model

A simplified example script for launch masked autoencoder pretraining is provided:
[pretrain.py](examples/train/pretrain.py). For DDP supported pretraining, check
[cinema/mae/pretrain.py](cinema/mae/pretrain.py). Check [examples/dicom_to_nifti.py](examples/dicom_to_nifti.py) for UKB
data preprocessing.

```bash
python examples/train/pretrain.py
```

## References

The code is built upon several open-source projects:

- [UK Biobank Cardiac Preprocessing](https://github.com/baiwenjia/ukbb_cardiac)
- [Masked Autoencoders](https://github.com/facebookresearch/mae)
- [ConvMAE](https://github.com/Alpha-VL/ConvMAE)
- [MultiMAE](https://github.com/EPFL-VILAB/MultiMAE)
- [MONAI](https://github.com/Project-MONAI/MONAI)
- [PyTorch Vision](https://github.com/pytorch/vision)
- [PyTorch Image Models](https://github.com/huggingface/pytorch-image-models)

## Acknowledgement

## Contact

For any questions or suggestions, please [create an issue](https://github.com/mathpluscode/CineMA/issues/new).

For collaborations, please contact Yunguan Fu (yunguan.fu.18@ucl.ac.uk).

## Citation
