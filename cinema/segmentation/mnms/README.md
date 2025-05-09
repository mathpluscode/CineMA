# Multi-Centre, Multi-Vendor & Multi-Disease Cardiac Image Segmentation Challenge (M&Ms)

## Preprocessing

Read the [preprocessing readme](../../data/mnms/README.md) on how to preprocess the data.

## Training

The following command can be used to train/fine-tune the model on the dataset (please adjust the `model.ckpt_path` and
`data.dir` accordingly):

```bash
# unet from scratch
mnms_seg_train data.dir=~/.cache/cinema_datasets/mnms/processed model.name=unet
# from scratch
mnms_seg_train data.dir=~/.cache/cinema_datasets/mnms/processed model.name=convunetr
# fine-tune
mnms_seg_train data.dir=~/.cache/cinema_datasets/mnms/processed model.name=convunetr model.ckpt_path=
```

## Evaluation

The following command can be used to evaluate the model (please adjust the `model.ckpt_path` accordingly):

```bash
mnms_seg_eval --data_dir ~/.cache/cinema_datasets/mnms/processed --ckpt_path
```
