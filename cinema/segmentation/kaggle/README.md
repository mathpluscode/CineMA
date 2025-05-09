# Kaggle Second Annual Data Science Bowl

## Preprocessing

Read the [preprocessing readme](../../data/kaggle/README.md) on how to preprocess the data.

## Evaluation of Segmentation Models

Although Kaggle does not provide the ground truth segmentation masks, we can still evaluate the segmentation models and
compare the segmentation-derived ejection fraction (EF) with the ground truth EF. The predicted segmentation can also be
used to extract the frames of ED and ES phases used for EF calculation.

```bash
kaggle_seg_eval --data_dir ~/.cache/cinema_datasets/kaggle/processed --ckpt_path
```

### Evaluation

The following command can be used to evaluate the model (please adjust the `model.ckpt_path` accordingly):

```bash
kaggle_clf_eval --data_dir --ckpt_path
```
