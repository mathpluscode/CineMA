"""ResNet model.

https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
https://github.com/kenshohara/3D-ResNets-PyTorch/blob/master/models/resnet.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock as BasicBlock2d
from torchvision.models.resnet import Bottleneck as Bottleneck2d

if TYPE_CHECKING:
    from collections.abc import Callable


def conv3x3x3(in_chans: int, out_chans: int, stride: int = 1) -> nn.Module:
    """3x3x3 convolution with padding.

    Args:
        in_chans: number of input channels.
        out_chans: number of output channels.
        stride: stride of the convolution.

    Returns:
        3x3x3 convolution layer.
    """
    return nn.Conv3d(in_chans, out_chans, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1x1(in_chans: int, out_chans: int, stride: int = 1) -> nn.Module:
    """1x1x1 convolution with padding.

    Args:
        in_chans: number of input channels.
        out_chans: number of output channels.
        stride: stride of the convolution.

    Returns:
        1x1x1 convolution layer.
    """
    return nn.Conv3d(in_chans, out_chans, kernel_size=1, stride=stride, bias=False)


class BasicBlock3d(nn.Module):
    """Basic block for ResNet."""

    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        norm_layer: Callable[..., nn.Module] | None = None,
    ) -> None:
        """Initialize the basic block.

        Args:
            inplanes: number of input channels.
            planes: number of output channels.
            stride: stride of the convolution.
            downsample: downsample layer.
            norm_layer: normalization layer.
        """
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the basic block.

        Args:
            x: input tensor, (batch, in_chans, x, y, z).

        Returns:
            Output tensor, (batch, out_chans, x, y, z).
        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck3d(nn.Module):
    """Bottleneck block for ResNet."""

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        norm_layer: Callable[..., nn.Module] | None = None,
    ) -> None:
        """Initialize the bottleneck block.

        Args:
            inplanes: number of input channels.
            planes: number of output channels.
            stride: stride of the convolution.
            downsample: downsample layer.
            norm_layer: normalization layer.
        """
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self.conv1 = conv1x1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the bottleneck block.

        Args:
            x: input tensor, (batch, in_chans, x, y, z).

        Returns:
            Output tensor, (batch, out_chans * expansion, x, y, z).
        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet3d(nn.Module):
    """3D ResNet model."""

    def __init__(
        self,
        block: type[BasicBlock3d | Bottleneck3d],
        in_channels: int,
        num_classes: int,
        layers: list[int],
        layer_inplanes: list[int],
        norm_layer: Callable[..., nn.Module] | None = None,
    ) -> None:
        """Initialize the ResNet model.

        Args:
            block: block module.
            in_channels: number of input channels.
            num_classes: number of output channels.
            layers: number of blocks in each layer.
            layer_inplanes: number of channels in each layer.
            norm_layer: normalization layer.
        """
        super().__init__()
        if len(layers) != 4:
            raise ValueError(f"layers should have length 4, got {len(layers)}.")
        if len(layer_inplanes) != 4:
            raise ValueError(f"layer_inplanes should have length 4, got {len(layer_inplanes)}.")
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer

        self.inplanes = layer_inplanes[0]
        self.conv1 = nn.Conv3d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, layer_inplanes[0], layers[0])
        self.layer2 = self._make_layer(block, layer_inplanes[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, layer_inplanes[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, layer_inplanes[3], layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(layer_inplanes[3] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self, block: type[BasicBlock3d | Bottleneck3d], planes: int, blocks: int, stride: int = 1
    ) -> nn.Sequential:
        """Make a layer of blocks and update in_chans.

        Args:
            block: block module.
            planes: number of output channels.
            blocks: number of blocks.
            shortcut_type: shortcut type.
            stride: stride of the convolution.

        Returns:
            A layer of blocks.
        """
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample, norm_layer=norm_layer)]
        self.inplanes = planes * block.expansion
        layers += [block(self.inplanes, planes, norm_layer=norm_layer) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, image_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass of the ResNet model.

        Args:
            image_dict: dictionary of input images, (batch, in_chans, x, y, z).

        Returns:
            Output tensor, (batch, n_classes).
        """
        if len(image_dict) != 1:
            raise ValueError(f"image_dict should have length 1, got {len(image_dict)}.")
        view = next(iter(image_dict.keys()))
        x = image_dict[view]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def get_resnet3d(
    depth: int,
    in_chans: int,
    out_chans: int,
    layer_inplanes: list[int],
) -> ResNet3d:
    """Get a 3D ResNet model.

    Args:
        depth: depth of the ResNet.
        in_chans: number of input channels.
        out_chans: number of output channels.
        layer_inplanes: number of channels in each layer.

    Returns:
        A ResNet model.
    """
    conv_n_blocks = {
        10: [1, 1, 1, 1],
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
        200: [3, 24, 36, 3],
    }[depth]
    return ResNet3d(
        block=BasicBlock3d if depth < 50 else Bottleneck3d,
        layers=conv_n_blocks,
        layer_inplanes=layer_inplanes,
        in_channels=in_chans,
        num_classes=out_chans,
    )


class ResNet2d(ResNet):
    """2D ResNet model.

    Modifications are:
    - Allow to specify the number of input channels, using `in_channels`.
    - Allow to specify the channels in each layer, using `layer_inplanes`.
    """

    def __init__(  # noqa: C901
        self,
        block: type[BasicBlock2d | Bottleneck2d],
        in_channels: int,
        num_classes: int,
        layers: list[int],
        layer_inplanes: list[int],
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: list[bool] | None = None,
        norm_layer: Callable[..., nn.Module] | None = None,
    ) -> None:
        """Initialize the ResNet model.

        Args:
            block: block module.
            in_channels: number of input channels.
            num_classes: number of output channels.
            layers: number of blocks in each layer.
            layer_inplanes: number of channels in each layer.
            zero_init_residual: zero initialize the last BN in each residual branch.
            groups: number of groups for the 3x3 convolution.
            width_per_group: number of channels per group.
            replace_stride_with_dilation: replace stride with dilation.
            norm_layer: normalization layer.
        """
        super(ResNet, self).__init__()  # pylint: disable=bad-super-call
        if len(layers) != 4:
            raise ValueError(f"layers should have length 4, got {len(layers)}.")
        if len(layer_inplanes) != 4:
            raise ValueError(f"layer_inplanes should have length 4, got {len(layer_inplanes)}.")

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = layer_inplanes[0]
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                f"replace_stride_with_dilation should be None or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, layer_inplanes[0], layers[0])
        self.layer2 = self._make_layer(
            block, layer_inplanes[1], layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, layer_inplanes[2], layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, layer_inplanes[3], layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(layer_inplanes[3] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d | nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck2d) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock2d) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, image_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass of the ResNet model.

        Args:
            image_dict: dictionary of input images, (batch, in_chans, x, y).

        Returns:
            Output tensor, (batch, n_classes).
        """
        if len(image_dict) != 1:
            raise ValueError(f"image_dict should have length 1, got {len(image_dict)}.")
        view = next(iter(image_dict.keys()))
        x = image_dict[view]
        return super().forward(x)


def get_resnet2d(
    depth: int,
    in_chans: int,
    out_chans: int,
    layer_inplanes: list[int],
) -> ResNet2d:
    """Get a 2D ResNet model.

    Args:
        depth: depth of the ResNet.
        in_chans: number of input channels.
        out_chans: number of output channels.
        layer_inplanes: number of channels in each layer.

    Returns:
        A ResNet model.
    """
    conv_n_blocks = {
        10: [1, 1, 1, 1],
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
        200: [3, 24, 36, 3],
    }[depth]
    return ResNet2d(
        block=BasicBlock2d if depth < 50 else Bottleneck2d,
        layers=conv_n_blocks,
        layer_inplanes=layer_inplanes,
        in_channels=in_chans,
        num_classes=out_chans,
    )
