"""MYOPS20202 dataset.

https://zmiclab.github.io/zxh/0/myops20/

In label, left ventricular (LV) blood pool (labelled 500),
right ventricular blood pool (600), LV normal myocardium (200),
LV myocardial edema (1220), and LV myocardial scars (2221) have been provided.
"""

from cinema import UKB_SAX_SLICE_SIZE

MYOPS2020_SLICE_SIZE = UKB_SAX_SLICE_SIZE
MYOPS2020_SPACING = (1.0, 1.0, 10.0)
MYOPS2020_LABEL_MAP = {
    600: 1,  # RV blood pool
    500: 2,  # LV blood pool
    200: 3,  # LV normal myocardium
    1220: 4,  # LV myocardial edema
    2221: 5,  # LV myocardial scars
}
