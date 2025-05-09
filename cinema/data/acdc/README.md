# Automated Cardiac Diagnosis Challenge (ACDC)

The [Automated Cardiac Diagnosis Challenge (ACDC)](https://www.creatis.insa-lyon.fr/Challenge/acdc/) dataset consists of
100 training and 50 testing cine-MRI images, distributed among 5 different pathologies (see section III of the paper for
more details): normal subjects (NOR), myocardial infarction (MINF), dilated cardiomyopathy (DCM), hypertrophic
cardiomyopathy (HCM), and abnormal right ventricle (ARV). The data was acquired from one hospital in France over 6 years
using two different scanners/. The segmentaiton labels of LV, RV, and MYO are provided for ED and ES phases. The volume
and EF can thus be derived. Images are stored in NIfTI format with 12-35 time frames per patient. After resampling to
1mm x 1mm x 10mm, images have 5-11 slices.

<details>
<summary>Pathologies and criteria</summary>

| Abbreviation | Name                        | Criteria                                                                                                                |
| ------------ | --------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| NOR          | Normal/healthy subjects     | LV EF >50%; LV EDV <90mL/m2 for men and <80mL/m2 for women; wall thickness in diastole <12mm; RVV <100mL/m2; RV EF >40% |
| MINF         | Myocardial infarction       | LV EF <40%; and abnormal myocardial infarction.                                                                         |
| DCM          | Dilated cardiomyopathy      | LV EF <40%; LV >100mL/m2; wall thickness in diastole <12mm                                                              |
| HCM          | Hypertrophic cardiomyopathy | LV EF >55% (normal); wall thickness in diastole >15mm                                                                   |
| ARV          | Abnormal right ventricle    | RV >110mL/m2 for men and >100ml/m2 for women; RV EF <40%                                                                |

</details>

## Download Dataset

> [!NOTE]
>
> It is recommended to download the dataset to download the data under `~/.cache/cinema_datasets` as the integration
> tests uses this path. Otherwise define the path using environment variable `CINEMA_DATA_DIR`.
>
> For MacOS users, the `wget` command may not work. Please download the dataset manually and unzip it. Or install
> [brew](https://brew.sh) and run `brew install wget`.

The dataset can be downloaded from https://www.creatis.insa-lyon.fr/Challenge/acdc/.

```bash
wget -O acdc.zip https://humanheart-project.creatis.insa-lyon.fr/database/api/v1/folder/637218e573e9f0047faa00fc/download
UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip acdc.zip
```

After unzipping the file, the files will have the following structure:

```
acdc.zip
database/
├── MANDATORY_CITATION.md
├── training/
│   ├── patient001/
│   │   ├── Info.cfg
│   │   ├── MANDATORY_CITATION.md
│   │   ├── patient001_4d.nii.gz
│   │   ├── patient001_frame01.nii.gz
│   │   ├── patient001_frame01_gt.nii.gz
│   │   ├── patient001_frame12.nii.gz
│   │   ├── patient001_frame12_gt.nii.gz
│   ├── patient002/
│   ├── ...
├── testing/
│   ├── patient101/
│   ├── patient102/
│   ├── ...
```

The `Info.cfg` file contains the group, height, weight, number of frames, and the frame number of end-diastole and
end-systole for for each patient. For example, the file for patient 1 contains the following information:

```
ED: 1
ES: 12
Group: DCM
Height: 184.0
NbFrame: 30
Weight: 95.0
```

## Preprocessing

The preprocessing is performed on ED and ES images with the following steps:

- resampling the ED/ES SAX images to 1.0 x 1.0 x 10.0 mm.
- cropping the images to 192 x 192 for xy, based on the LV center at ED frame.
- normalizing the values to [0, 1].

```bash
acdc_preprocess
```

The CLI requires to install the `cinema` package at the root of repository with `pip install -e .`.
