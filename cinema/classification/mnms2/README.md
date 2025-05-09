# Multi-Disease, Multi-View & Multi-Center Right Ventricular Segmentation in Cardiac MRI (M&Ms2)

## Preprocessing

Read the [preprocessing readme](../../data/mnms2/README.md) on how to preprocess the data.

## Training

The following command can be used to train/fine-tune the model on the dataset (please adjust the `model.ckpt_path` and
`data.dir` accordingly):

```bash
# from scratch
mnms2_clf_train data.dir=~/.cache/cinema_datasets/mnms2/processed
# fine-tune
mnms2_clf_train data.dir=~/.cache/cinema_datasets/mnms2/processed model.ckpt_path=
```

## Evaluation

The following command can be used to evaluate the model (please adjust the `model.ckpt_path` accordingly):

```bash
mnms2_clf_eval --data_dir /cluster/project7/cardiac_71702_processed/mnms2/processed --ckpt_path
```
