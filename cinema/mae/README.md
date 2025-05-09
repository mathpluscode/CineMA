# MAE Pretraining using UK Biobank data

## Data preprocessing

Follow the provided [script](../examples/dicom_to_nifti.py) to process UK Biobank data. It is recommended to download
the dataset to store the data under `~/.cache/cinema_datasets/ukb` as the integration tests uses this path.

## Launch pre-training locally

For local debugging, you can use the following command.

```bash
ukb_mae_pretrain model.size=tiny train.batch_size=16
```
