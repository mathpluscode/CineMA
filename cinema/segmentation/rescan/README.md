# Scan-rescan dataset

## Preprocessing

Read the [preprocessing readme](../../data/rescan/README.md) on how to preprocess the data.

## Training

The following command can be used to train/fine-tune the model on the dataset (please adjust the `model.ckpt_path` and
`data.dir` accordingly):

```bash
# unet from scratch
rescan_seg_train data.dir=~/.cache/cinema_datasets/rescan/processed model.name=unet
# from scratch
rescan_seg_train data.dir=~/.cache/cinema_datasets/rescan/processed model.name=convunetr
# fine-tune
rescan_seg_train data.dir=~/.cache/cinema_datasets/rescan/processed model.name=convunetr model.ckpt_path=
```

## Evaluation

The following command can be used to evaluate the model (please adjust the `model.ckpt_path` accordingly):

```bash
rescan_seg_eval --data_dir ~/.cache/cinema_datasets/rescan/processed --ckpt_path
```
