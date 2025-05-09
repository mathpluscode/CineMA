"""Multi-Centre, Multi-Vendor & Multi-Disease Cardiac Image Segmentation Challenge (M&Ms).

https://www.ub.edu/mnms/

Original classes:
    LV, left ventricle, class 1
    MYO, myocardium, class 2
    RV, right ventricle, class 3

M&Ms has spacing 5-10mm on z-axis.
If resampling to 5mm, we will have 8-32 slices.
If resampling to 10mm, we will have 4-16 slices.
"""

from cinema import LV_LABEL, MYO_LABEL, RV_LABEL, UKB_LAX_SLICE_SIZE, UKB_SAX_SLICE_SIZE

MNMS_SPACING = (1.0, 1.0, 10.0)
MNMS_SAX_SLICE_SIZE = UKB_SAX_SLICE_SIZE
MNMS_LAX_SLICE_SIZE = UKB_LAX_SLICE_SIZE
MNMS_LABEL_MAP = {1: LV_LABEL, 2: MYO_LABEL, 3: RV_LABEL}
