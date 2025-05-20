# MAE Pretraining using UK Biobank data

## Data preprocessing

Follow the provided [script](../examples/dicom_to_nifti.py) to process UK Biobank data. It is recommended to download
the dataset to store the data under `~/.cache/cinema_datasets/ukb` as the integration tests uses this path. The data
structure would be

```
ukb/processed/
├── 6000182_2/
│   ├── 6000182_2_lax_2c.nii.gz
│   ├── 6000182_2_lax_3c.nii.gz
│   ├── 6000182_2_lax_4c.nii.gz
│   ├── 6000182_2_sax.nii.gz
├── 6000366_2/
├── ...
```

## Launch pre-training locally

For local debugging, you can use the following command.

```bash
ukb_mae_pretrain model.size=tiny train.batch_size=16
```
