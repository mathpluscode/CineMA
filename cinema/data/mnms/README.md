# Multi-Centre, Multi-Vendor & Multi-Disease Cardiac Image Segmentation Challenge (M&Ms)

> [!CAUTION]
>
> Only 345 images are available without restriction. The remaining 30 images require application.

The [M&Ms](https://www.ub.edu/mnms/) dataset contains 375 images (175 training, 40 validation and 160 testing) acquired
by four different scanner vendors in six hospitals across three countries. The pathologies include hypertrophic
cardiomyopathy (HCM), dilated cardiomyopathy (DCM), hypertensive heart disease (HHD), abnormal right ventricle (ARV),
athlete heart syndrome (AHS), ischemic heart disease (IHD), and left ventricle non-compaction (LVNC). The segmentation
labels of LV, RV, and MYO are provided for ED and ES phases.

## Download Dataset

> [!NOTE]
>
> It is recommended to download the dataset to download the data under `~/.cache/cinema_datasets` as the integration
> tests uses this path. Otherwise define the path using environment variable `CINEMA_DATA_DIR`.

Download the dataset from [the website](https://www.ub.edu/mnms/) and unzip the file.

```bash
unzip MnM.zip
```

After unzipping the file, the files will have the following structure:

```
MnM.zip
OpenDataset/
├── 211230_M&Ms_Dataset_information_diagnosis_opendataset.csv
├── Training/
│   ├── Labeled/
│   │   ├── A0S9V9/
│   │   │   ├── A0S9V9_sa_gt.nii.gz
│   │   │   ├── A0S9V9_sa.nii.gz
│   │   ├── ...
│   ├── Unlabeled/ # images without segmentation labels despite the presence of _gt.nii.gz files
├── Validation/
│   ├── A5C2D2/
│   ├── ...
├── Testing/
│   ├── A1K2P5/
```

## Preprocessing

The preprocessing is performed on ED and ES images with the following steps:

- resampling the ED/ES SAX images to 1.0 x 1.0 x 10.0 mm.
- cropping the images to 192 x 192 for xy, based on the LV center at ED frame.
- normalizing the values to [0, 1].

```bash
mnms_preprocess
```

The CLI requires to install the `cinema` package at the root of repository with `pip install -e .`.
