# MyoPS 2020: Myocardial pathology segmentation combining multi-sequence CMR

The [MyoPS2020](https://zmiclab.github.io/zxh/0/myops20/) dataset consists of 25 training and 20 testing subjects. Each
subject includes late gadolinium enhancement (LGE) cmr, the T2-weighted CMR, and the balanced- Steady State Free
Precession (bSSFP). The segmentaiton labels of left ventricular (LV) blood pool (labelled 500), right ventricular blood
pool (600), LV normal myocardium (200), LV myocardial edema (1220), and LV myocardial scars (2221) have been provided.
Images are stored in NIfTI format. After resampling to 1mm x 1mm x 10mm, images have 3 - 6 slices.

## Download Dataset

> [!NOTE]
>
> It is recommended to download the dataset to download the data under `~/.cache/cinema_datasets/myops2020` as the
> integration tests uses this path. Otherwise define the path using environment variable `CINEMA_DATA_DIR`.

The dataset can be downloaded from https://zmiclab.github.io/zxh/0/myops20/index.html.

After unzipping, the files will have the following structure:

```
train25/
├── myops_training_101_C0.nii.gz
├── myops_training_101_DE.nii.gz
├── myops_training_101_T2.nii.gz
├── myops_training_102_C0.nii.gz
├── myops_training_102_DE.nii.gz
├── myops_training_102_T2.nii.gz
├── ...
train25_myops_gd/
├── myops_training_101_gd.nii.gz
├── myops_training_102_gd.nii.gz
├── ...
test20/
├── myops_training_201_C0.nii.gz
├── myops_training_201_DE.nii.gz
├── myops_training_201_T2.nii.gz
├── myops_training_202_C0.nii.gz
├── myops_training_202_DE.nii.gz
├── myops_training_202_T2.nii.gz
├── ...
MyoPS2020_EvaluateByYouself/
```

## Preprocessing

The preprocessing is performed on ED and ES images with the following steps:

- resampling the ED/ES SAX images to 1.0 x 1.0 x 10.0 mm.
- cropping the images to 192 x 192 for xy, based on the LV center at ED frame.
- normalizing the values to [0, 1].

```bash
myops2020_preprocess
```

The CLI requires to install the `cinema` package at the root of repository with `pip install -e .`.
