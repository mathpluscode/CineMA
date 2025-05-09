# Multi-Disease, Multi-View & Multi-Center Right Ventricular Segmentation in Cardiac MRI (M&Ms2)

## Preprocessing

Read the [preprocessing readme](../../data/mnms2/README.md) on how to preprocess the data.

## Training

The following command can be used to train/fine-tune the model on the dataset (please adjust the `model.ckpt_path` and
`data.dir` accordingly):

```bash
# unet from scratch
mnms2_seg_train data.dir=~/.cache/cinema_datasets/mnms2/processed model.name=unet
# from scratch
mnms2_seg_train data.dir=~/.cache/cinema_datasets/mnms2/processed model.name=convunetr
# fine-tune
mnms2_seg_train data.dir=~/.cache/cinema_datasets/mnms2/processed model.name=convunetr model.ckpt_path=
```

## Evaluation

The following command can be used to evaluate the model (please adjust the `model.ckpt_path` accordingly):

```bash
mnms2_seg_eval --data_dir ~/.cache/cinema_datasets/mnms2/processed --ckpt_path
```
