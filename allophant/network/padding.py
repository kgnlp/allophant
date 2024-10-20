from typing import Tuple
import typing

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn import functional


def get_padding(kernel_size: int, stride: int = 1, stft_type: bool = False) -> Tuple[int, int]:
    if stft_type:
        padding = kernel_size // 2
        if stride == 1:
            return (padding, padding - 1)
        else:
            return (padding, padding)
    if stride > 1:
        # Adds enough padding for a filter with a higher stride to reach the edge
        return (kernel_size // 2, kernel_size - 1)
    padding = kernel_size // 2
    return (padding, padding)


class VariableLengthReflectPad(Module):
    def __init__(self, padding: Tuple[int, int]):
        super().__init__()
        self._padding = padding
        self._left_padding, self._right_padding = padding
        pad_base_indices = torch.arange(0, self._right_padding).view(1, 1, -1)
        right_pad_start_indices = pad_base_indices + self._left_padding
        right_pad_end_indices = pad_base_indices + 2
        left_pad_indices = torch.arange(self._left_padding, 0, -1)
        self.register_buffer("_right_pad_start_indices", right_pad_start_indices)
        self.register_buffer("_right_pad_end_indices", right_pad_end_indices)
        self.register_buffer("_left_pad_indices", left_pad_indices)

    @property
    def padding(self) -> Tuple[int, int]:
        return self._padding

    def forward(self, features: Tensor, lengths: Tensor) -> Tensor:
        padded = functional.pad(features, self._padding)
        feature_size = features.size(1)
        padded[..., : self._left_padding] = features.gather(
            -1, typing.cast(Tensor, self._left_pad_indices).repeat(1, feature_size, 1)
        )
        lengths = lengths.view(-1, 1, 1)
        padded.scatter_(
            -1,
            (lengths + self._right_pad_start_indices).repeat(1, feature_size, 1),
            features.gather(-1, (lengths - self._right_pad_end_indices).repeat(1, feature_size, 1)),
        )
        return padded
