# Kaggle Second Annual Data Science Bowl

The [Kaggle cardiac MRI dataset](https://www.kaggle.com/c/second-annual-data-science-bowl/data) dataset consists of 1140
cine-MRI images, split into 500 training, 200 validation, and 440 test images. The images are stored in DICOM format.
Only the left ventricle volume for the ED and ES phases is provided. Labels on test images are not available. The EF can
be derived. In the original competition, the Continuous Ranked Probability Score (CRPS) is used for evaluation.

## Download

> [!NOTE]
>
> It is recommended to download the dataset to download the data under `~/.cache/cinema_datasets` as the integration
> tests uses this path. Otherwise define the path using environment variable `CINEMA_DATA_DIR`.
>
> If you see `403 - Forbidden - You must accept this competition's rules before you'll be able to download files` error,
> you need to join the competition and accept the rules on the Kaggle website. You may need to verify your Kaggle
> account.

This data set can be downloaded from [Kaggle](https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1) using
[kaggle API](https://www.kaggle.com/docs/api).

1.  The [authentication token](https://www.kaggle.com/docs/api#getting-started-installation-&-authentication) shall be
    obtained and stored under `~/.kaggle/kaggle.json`. Run the following command to change the permissions.

        ```bash
        chmod 600 ~/.kaggle/kaggle.json
        ```

2.  Then, execute the following commands to download and unzip files, the `unzip` command will not print any output.

    ```bash
    pip install kaggle
    kaggle competitions download -c second-annual-data-science-bowl
    unzip -q second-annual-data-science-bowl.zip -d second-annual-data-science-bowl
    ```

## Preprocessing

> [!WARNING]
>
> DICOM images contain lots of files. This may hit the limit of the number of files that can be stored.

Use the `kaggle_preprocess` entrypoint to preprocess the data with the following steps:

- resampling LAX 2C/4C images to 1.0 x 1.0 mm and SAX images to 1.0 x 1.0 x 10.0 mm.
- cropping the images to 256 x 256 for LAX and 192 x 192 for SAX, based on 2C/4C/SAX intersection center.
- normalizing the values to [0, 1].

```bash
kaggle_preprocess
```

The CLI requires to install the `cinema` package at the root of repository with `pip install -e .`.
