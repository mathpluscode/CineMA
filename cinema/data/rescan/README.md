# Rescan

> For access to the dataset, please contact Rhodri Davies (rhodri.davies@ucl.ac.uk).

The [Rescan](https://www.ahajournals.org/doi/10.1161/CIRCIMAGING.119.009214) dataset was obtained from 5 United Kingdom
institutions with 6 different MRI scanners of 2 field strengths (1.5T, 3T), 2 manufacturers (Siemens, Philips), and 3
models (Aera, Achieva, Avanto). The patients characteristics include healthy, myocardial infarction, left ventricular
hypertrophy, cardiomyopathy, and other pathologies. For the same subject, two scans were acquired within a short time
frame and thus the cardiac function is assumed to be the same. The objective is to evaluate machine learning models's
variance in predicting EF, using [coefficients of variation (CV)](https://www-users.york.ac.uk/~mb55/meas/cv.htm) as the
metric.

There are three sources of data

- train: unpaired data with segmentation labels derived from a segmentation model trained separately
- test: paired data with segmentation labels derived from a segmentation model trained separately
- test_retest_100: paired data without segmentation labels, but with EDV/ESV values

## Preprocess

> [!NOTE]
>
> It is recommended to download the dataset to download the data under `~/.cache/cinema_datasets` as the integration
> tests uses this path. Otherwise define the path using environment variable `CINEMA_DATA_DIR`.

The files should be stored in the following structure:

```
rescan/
├── pickle/
│   ├── train/
│   │   ├── A/
│   │   │   ├── s_0000/
│   │   │   │   ├── 2C.pickle
│   │   │   │   ├── 4C.pickle
│   │   │   │   ├── SAX.pickle
│   │   │   │   ├── SAX_segs.pickle
│   │   │   ├── s_0001/
│   │   │   ├── ...
│   │   ├── B/
│   │   ├── ...
│   ├── test/
│   │   ├── scan_00_A/
│   │   │   ├── 2C.pickle
│   │   │   ├── 4C.pickle
│   │   │   ├── SAX.pickle
│   │   │   ├── SAX_segs.pickle
│   │   ├── scan_00_B/
│   │   ├── ...
│   ├── test_retest_100/
│   │   ├── 1027/
│   │   │   │   ├── 2C.pickle
│   │   │   │   ├── 4C.pickle
│   │   │   │   ├── SAX.pickle
│   │   ├── 1153/
│   │   │   │   ├── 2C.pickle
│   │   │   │   ├── 4C.pickle
│   │   │   │   ├── SAX.pickle
│   │   ├── ...
├── processed/
│   ├── train/
│   │   ├── A/
│   │   │   ├── s_0000/
│   │   │   │   ├── lax_2c_t.nii.gz
│   │   │   │   ├── lax_4c_t.nii.gz
│   │   │   │   ├── sax_t.nii.gz
│   │   │   │   ├── sax_gt_t.nii.gz

```

The preprocessing is performed on SAX and LAX images with the following steps:

- resampling the SAX images to 1.0 x 1.0 x 10.0 mm and LAX images to 1.0 x 1.0 mm.
- cropping the images to SAX, LAX image intersection.
- normalizing the values to [0, 1].

```commandline
rescan_preprocess --data_dir pickle --out_dir processed
```
