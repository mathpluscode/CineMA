"""Kaggle dataset.

https://www.kaggle.com/c/second-annual-data-science-bowl/data

Kaggle mostly have 30 frames, some have 25 and some have 60 or 90.
Kaggle has spacing 5-10mm on z-axis.
If resampling to 5mm, we will have 5-30 slices.
If resampling to 10mm, we will have 2-15 slices.

The dataset has diastole and systole volumes, and we can calculate ejection fraction.
Diastole volume is between 10-569ml.
Systole volume is between 4-489ml.
Ejection fraction is
- between 9-78% on train split, average is ~58%.
- between 19-76% on val split.
- between 15-83% on test split.

Often the middle point of LVEF is considered as 60%, see:
https://www.ncbi.nlm.nih.gov/books/NBK459131/
"""

from cinema import UKB_LAX_SLICE_SIZE, UKB_SAX_SLICE_SIZE

KAGGLE_SPACING = (1.0, 1.0, 10.0)
KAGGLE_SAX_SLICE_SIZE = UKB_SAX_SLICE_SIZE
KAGGLE_LAX_SLICE_SIZE = UKB_LAX_SLICE_SIZE
