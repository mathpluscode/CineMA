# Automated Cardiac Diagnosis Challenge (ACDC)

## Preprocessing

Read the [preprocessing readme](../../data/acdc/README.md) on how to preprocess the data.

## Training

The following command can be used to train/fine-tune the model on the dataset (please adjust the `model.ckpt_path` and
`data.dir` accordingly):

```bash
# from scratch
acdc_clf_train data.dir=~/.cache/cinema_datasets/acdc/processed
# fine-tune
acdc_clf_train data.dir=~/.cache/cinema_datasets/acdc/processed model.ckpt_path=
```

## Evaluation

The following command can be used to evaluate the model (please adjust the `model.ckpt_path` accordingly):

```bash
acdc_clf_eval --data_dir ~/.cache/cinema_datasets/acdc/processed --ckpt_path
```
