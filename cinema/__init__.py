"""CaridaX package."""

from cinema.convvit import ConvViT
from cinema.mae.mae import CineMA
from cinema.metric import heatmap_soft_argmax
from cinema.segmentation.convunetr import ConvUNetR
from cinema.vit import patchify, unpatchify

UKB_SPACING = (1.0, 1.0, 10.0)
UKB_LAX_SLICE_SIZE = (256, 256)
UKB_SAX_SLICE_SIZE = (192, 192)
UKB_N_FRAMES = 50  # UKB has always 50 frames

RV_LABEL = 1
MYO_LABEL = 2
LV_LABEL = 3
LABEL_TO_NAME = {
    RV_LABEL: "RV",
    MYO_LABEL: "MYO",
    LV_LABEL: "LV",
}

__all__ = [
    "LABEL_TO_NAME",
    "LV_LABEL",
    "MYO_LABEL",
    "RV_LABEL",
    "CineMA",
    "ConvUNetR",
    "ConvViT",
    "heatmap_soft_argmax",
    "patchify",
    "unpatchify",
]
