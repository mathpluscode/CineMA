# Landmark dataset

> For access to the dataset, please contact Rhodri Davies (rhodri.davies@ucl.ac.uk).

## Download Dataset

> [!NOTE]
>
> It is recommended to download the dataset to `~/.cache/cinema_datasets/landmark` as the integration tests are based on
> this path.

The files should have the following structure:

```
lax_2c.csv
lax_2c/
├── images/
│   ├── 0001__00.png
│   ├── 0001__09.png
│   ├── ...
├── masks/
│   ├── 0001__00.png
│   ├── 0001__09.png
│   ├── ...
lax_4c.csv
lax_4c/
├── images/
│   ├── 0001__00.png
│   ├── 0001__09.png
│   ├── ...
├── masks/
│   ├── 0001__00.png
│   ├── 0001__09.png
│   ├── ...
```

## Training

The following command can be used to train/fine-tune the model on the dataset (please adjust the `model.ckpt_path` and
`data.dir` accordingly):

```bash
# from scratch
landmark_reg_train data.dir=~/.cache/cinema_datasets/landmark/processed
# fine-tune
landmark_reg_train data.dir=~/.cache/cinema_datasets/landmark/processed model.ckpt_path=
```

## Evaluation

The following command can be used to evaluate the model (please adjust the `model.ckpt_path` accordingly):

```bash
landmark_reg_eval --data_dir ~/.cache/cinema_datasets/landmark/processed --ckpt_path
```
