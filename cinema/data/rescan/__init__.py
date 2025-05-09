"""Scan-rescan data.

https://www.ahajournals.org/doi/10.1161/CIRCIMAGING.119.009214

Original classes:
    LV, left ventricle, class 1
    MYO, myocardium, class 2
    RV, right ventricle, class 3

Images have 10-30 frames, average 25 frames.
Rescan SAX has spacing 8-10mm on z-axis.
If resampling to 5mm, we will have 16-36 slices.
If resampling to 10mm, we will have 8-18 slices.
"""

from cinema import LV_LABEL, MYO_LABEL, RV_LABEL, UKB_LAX_SLICE_SIZE, UKB_SAX_SLICE_SIZE

RESCAN_SPACING = (1.0, 1.0, 10.0)
RESCAN_SAX_SLICE_SIZE = UKB_SAX_SLICE_SIZE
RESCAN_LAX_SLICE_SIZE = UKB_LAX_SLICE_SIZE
RESCAN_LABEL_MAP = {1: LV_LABEL, 2: MYO_LABEL, 3: RV_LABEL}
