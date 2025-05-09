"""Multi-Disease, Multi-View & Multi-Center Right Ventricular Segmentation in Cardiac MRI (M&Ms2).

https://www.ub.edu/mnms-2/

Original classes:
    LV, left ventricle, class 1
    MYO, myocardium, class 2
    RV, right ventricle, class 3

M&Ms2 has spacing 8-20mm on z-axis.
If resampling to 5mm, we will have 12-34 slices.
If resampling to 10mm, we will have 6-17 slices.
"""

from cinema import LV_LABEL, MYO_LABEL, RV_LABEL, UKB_LAX_SLICE_SIZE, UKB_SAX_SLICE_SIZE

MNMS2_SPACING = (1.0, 1.0, 10.0)
MNMS2_SAX_SLICE_SIZE = UKB_SAX_SLICE_SIZE
MNMS2_LAX_SLICE_SIZE = UKB_LAX_SLICE_SIZE
MNMS2_LABEL_MAP = {1: LV_LABEL, 2: MYO_LABEL, 3: RV_LABEL}
