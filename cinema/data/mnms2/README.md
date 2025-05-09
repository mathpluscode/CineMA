# Multi-Disease, Multi-View & Multi-Center Right Ventricular Segmentation in Cardiac MRI (M&Ms2)

> [!CAUTION]
>
> The pathology classes RV, TRI do not exist in training split, while exist in validation and test splits.

The [M&Ms2](https://www.ub.edu/mnms-2/) dataset contains 360 images (160 training, 40 validation and 160 testing)
acquired from three clinical centers in Spain with 8 different scanners from 3 vendors in six hospitals across three
countries. The pathologies include dilated left ventricle (DLV), dilated right ventricle (DRV), hypertrophic
cardiomyopathy (HCM), arrhythmogenic cardiomyopathy (ARR), tetrology of fallot (FALL), inter-atrial communication (CIA),
and tricuspid regurgitation (TRI). See the [paper](https://ieeexplore.ieee.org/document/10103611) for more details. The
segmentation labels of LV, RV, and MYO are provided for ED and ES phases for short-axis and long-axis four-chamber
views. The images are stored in NIfTI format.

<details>
<summary>Pathologies and criteria</summary>

| Abbreviation | Name                          | Criteria                                                                                                                                                            |
| ------------ | ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| DLV          | Dilated left ventricle        | LV EDV >214mL(>105mL/m2) for men and >179 mL(>96mL/m2) for women                                                                                                    |
| RV           | Dilated right ventricle       | RV EDV >250mL(>121mL/m2) for men and >201 mL(>112mL/m2) for women                                                                                                   |
| HCM          | Hypertrophic cardiomyopathy   | LV wall thickness >15mm                                                                                                                                             |
| ARR          | Arrhythmogenic cardiomyopathy | global RV dilatation and wall motion abnormalities with or without a decreased EF                                                                                   |
| FALL         | Tetrology of fallot           | a nonrestrictive ventricular septal defect, overriding aorta; right ventricle outflow tract obstruction and/or branch pulmonary artery stenosis; and RV hypertrophy |
| CIA          | Inter-atrial communication    | RV volume overload, identification of inferior sinus venous defect in the long-axis 4-chamber view                                                                  |
| TRI          | Tricuspid regurgitation       | one or more flow jets emanating from the tricuspid valve and projecting into the RV, often holosystolic and readily apparent on the long-axis 4-chamber view        |

</details>

## Download Dataset

> [!NOTE]
>
> It is recommended to download the dataset to download the data under `~/.cache/cinema_datasets` as the integration
> tests uses this path. Otherwise define the path using environment variable `CINEMA_DATA_DIR`.

Download the dataset from [the website](https://www.ub.edu/mnms-2/) and unzip the `.rar` file.

After unzipping the file, the files will have the following structure. Note, 160 training subjects, 40 validation
subjects and 160 testing subjects are ordered sequentially.

```
MnM2.zip
MnM2/
├── dataset_information.csv
├── dataset/
│   ├── 001/
│   │   ├── 001_LA_CINE.nii.gz
│   │   ├── 001_LA_ED_gt.nii.gz
│   │   ├── 001_LA_ED.nii.gz
│   │   ├── 001_LA_ES_gt.nii.gz
│   │   ├── 001_LA_ES.nii.gz
│   │   ├── 001_SA_CINE.nii.gz
│   │   ├── 001_SA_ED_gt.nii.gz
│   │   ├── 001_SA_ED.nii.gz
│   │   ├── 001_SA_ES_gt.nii.gz
│   │   ├── 001_SA_ES.nii.gz
│   ├── 002/
│   ├── ...
```

## Preprocessing

The preprocessing is performed on ED and ES images with the following steps:

- resampling LAX 4C images to 1.0 x 1.0 mm and SAX images to 1.0 x 1.0 x 10.0 mm.
- cropping the images to 256 x 256 for LAX and 192 x 192 for SAX, based on LV center at ED frame.
- normalizing the values to [0, 1].

```bash
mnms2_preprocess
```

The CLI requires to install the `cinema` package at the root of repository with `pip install -e .`.
