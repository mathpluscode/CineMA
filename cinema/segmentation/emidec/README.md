# EMIDEC

## Preprocessing

Read the [preprocessing readme](../../data/emidec/README.md) on how to preprocess the data.

## Training

The following command can be used to train/fine-tune the model on the dataset (please adjust the `model.ckpt_path` and
`data.dir` accordingly):

```bash
# unet from scratch
emidec_seg_train data.dir=~/.cache/cinema_datasets/emidec/processed model.name=unet
# from scratch
emidec_seg_train data.dir=~/.cache/cinema_datasets/emidec/processed model.name=convunetr
# fine-tune
emidec_seg_train data.dir=~/.cache/cinema_datasets/emidec/processed model.name=convunetr model.ckpt_path=
```

## Evaluation

The following command can be used to evaluate the model (please adjust the `model.ckpt_path` accordingly):

```bash
emidec_seg_eval --data_dir ~/.cache/cinema_datasets/emidec/processed --ckpt_path
```
