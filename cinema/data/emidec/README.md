# EMIDEC automatic Evaluation of Myocardial Infarction from Delayed-Enhancement Cardiac MRI

The [EMIDEC](https://emidec.com/) dataset consists of 100 training (33 normal cases and 67 pathological cases) and 50
testing DE-MRI images. Labels on test set are not provided. The segmentaiton labels of myocardium, cavity, myocardial
infarction and no-reflow are provided. Images are stored in NIfTI format. After resampling to 1mm x 1mm x 10mm, images
have 5-10 slices.

## Download Dataset

> [!NOTE]
>
> It is recommended to download the dataset to download the data under `~/.cache/cinema_datasets/emidec` as the
> integration tests uses this path. Otherwise define the path using environment variable `CINEMA_DATA_DIR`.

The dataset can be downloaded from https://emidec.com/.

After unzipping, the files will have the following structure:

```
emidec-dataset-1.0.1.zip
emidec-segmentation-testset-1.0.0.zip
emidec-dataset-1.0.1/
├── Readme.txt
├── Case N006.txt
├── Case N012.txt
├── ...
├── Case_N006/
│   ├── Contours/
│   │   └── Case_N006.nii.gz
│   ├── Images/
│       └── Case_N006.nii.gz
├── Case_N012/
│   ├── Contours/
│   │   └── Case_N006.nii.gz
│   ├── Images/
│       └── Case_N006.nii.gz
├── ...
```

In the case name, `N` stands for normal, `M` for myocardial infarction, `P` for pathological. The `Case N006.txt` file
contains the sex, age, and other information about the patient.

```
Case N006

Gap between slices†: 10 mm
Sex†: F
Age : 70
Tobacco : 3
Overweight : N
Arterial hypertension : Y
Diabetes : N
Familial history of coronary artery disease: N
ECG (ST +) : Y
Troponin : 1.1
Killip Max: 1
FEVG : 60
NTProBNP : 248
```

## Preprocessing

The preprocessing is performed with the following steps:

- resampling images to 1.0 x 1.0 x 10.0 mm.
- cropping the images to 192 x 192 for xy, based on the cavity center.
- normalizing the values to [0, 1].

```bash
emidec_preprocess
```

The CLI requires to install the `cinema` package at the root of repository with `pip install -e .`.
