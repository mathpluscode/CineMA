# Landmark dataset

> For access to the dataset, please contact Rhodri Davies (rhodri.davies@ucl.ac.uk).

## Download Dataset

> [!NOTE]
>
> It is recommended to download the dataset to download the data under `~/.cache/cinema_datasets` as the integration
> tests uses this path. Otherwise define the path using environment variable `CINEMA_DATA_DIR`.

The files should have the following structure:

```
raw/
├── lax_2c.csv
├── lax_2c/
│   ├── images/
│   │   ├── 0001__00.png
│   │   ├── 0001__09.png
│   │   └── ...
│   └── masks/
│       ├── 0001__00.png
│       ├── 0001__09.png
│       └── ...
├── lax_4c.csv
├── lax_4c/
│   ├── images/
│   │   ├── 0001__00.png
│   │   ├── 0001__09.png
│   │   └── ...
│   └── masks/
│       ├── 0001__00.png
│       ├── 0001__09.png
│       └── ...
```

To preprocess

```commandline
landmark_preprocess --data_dir raw --out_dir processed
```

The CLI requires to install the `cinema` package at the root of repository with `pip install -e .`.
