# Multi-Centre, Multi-Vendor & Multi-Disease Cardiac Image Segmentation Challenge (M&Ms)

## Preprocessing

Read the [preprocessing readme](../../data/mnms/README.md) on how to preprocess the data.

## Training

The following command can be used to train/fine-tune the model on the dataset (please adjust the `model.ckpt_path` and
`data.dir` accordingly):

```bash
# from scratch
mnms_reg_train data.dir=~/.cache/cinema_datasets/mnms/processed
# fine-tune
mnms_reg_train data.dir=~/.cache/cinema_datasets/mnms/processed model.ckpt_path=
```

## Evaluation

The following command can be used to evaluate the model (please adjust the `model.ckpt_path` accordingly):

```bash
mnms_reg_eval --data_dir ~/.cache/cinema_datasets/mnms/processed --ckpt_path
```
