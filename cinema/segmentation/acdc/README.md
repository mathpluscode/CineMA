# Automated Cardiac Diagnosis Challenge (ACDC)

## Preprocessing

Read the [preprocessing readme](../../data/acdc/README.md) on how to preprocess the data.

## Training

The following command can be used to train/fine-tune the model on the dataset (please adjust the `model.ckpt_path` and
`data.dir` accordingly):

```bash
# unet from scratch
acdc_seg_train data.dir=~/.cache/cinema_datasets/acdc/processed model.name=unet
# from scratch
acdc_seg_train data.dir=~/.cache/cinema_datasets/acdc/processed model.name=convunetr
# fine-tune
acdc_seg_train data.dir=~/.cache/cinema_datasets/acdc/processed model.name=convunetr model.ckpt_path=
```

## Evaluation

The following command can be used to evaluate the model (please adjust the `model.ckpt_path` accordingly):

```bash
acdc_seg_eval --data_dir ~/.cache/cinema_datasets/acdc/processed --ckpt_path
```
