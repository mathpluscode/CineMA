"""EMIDEC dataset.

https://emidec.com/
https://github.com/EMIDEC-Challenge/Evaluation-metrics/blob/master/main.py

0: background
1: cavity
2: myocardium
3: myocardial infarction
4: no-reflow

For evaluation
- cavity: 1
- myocardium: 2+3+4
- myocardial infarction: 3+4
- no-reflow: 4
"""

from cinema import UKB_SAX_SLICE_SIZE

EMIDEC_SLICE_SIZE = UKB_SAX_SLICE_SIZE
EMIDEC_SPACING = (1.458, 1.458, 10.0)  # https://emidec.com/downloads/papers/paper-16.pdf
