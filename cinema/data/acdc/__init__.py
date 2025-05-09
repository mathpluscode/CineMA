"""ACDC MR image dataset.

https://www.creatis.insa-lyon.fr/Challenge/acdc/

Original classes:
- RV: right ventricle, 1
- MYO: myocardium, 2
- LV: left ventricle, 3

ACDC has spacing 5-10mm on z-axis.
If resampling to 5mm, we will have 10-22 slices.
If resampling to 10mm, we will have 5-11 slices.
"""

from cinema import LV_LABEL, MYO_LABEL, RV_LABEL, UKB_SAX_SLICE_SIZE

ACDC_SAX_SLICE_SIZE = UKB_SAX_SLICE_SIZE
ACDC_SPACING = (1.0, 1.0, 10.0)
ACDC_LABEL_MAP = {3: LV_LABEL, 2: MYO_LABEL, 1: RV_LABEL}
