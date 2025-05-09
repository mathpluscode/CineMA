# MYOPS2020

## Preprocessing

Read the [preprocessing readme](../../data/myops2020/README.md) on how to preprocess the data.

## Training

The following command can be used to train/fine-tune the model on the dataset (please adjust the `model.ckpt_path` and
`data.dir` accordingly):

```bash
# unet from scratch
myops2020_seg_train data.dir=~/.cache/cinema_datasets/myops2020/processed model.name=unet
# from scratch
myops2020_seg_train data.dir=~/.cache/cinema_datasets/myops2020/processed model.name=convunetr
# fine-tune
myops2020_seg_train data.dir=~/.cache/cinema_datasets/myops2020/processed model.name=convunetr model.ckpt_path=
```

## Evaluation

The following command can be used to evaluate the model (please adjust the `model.ckpt_path` accordingly):

```bash
myops2020_seg_eval --data_dir ~/.cache/cinema_datasets/myops2020/processed --ckpt_path
```
