from typing import TypeVar, Type, Tuple, Callable, Optional, Sequence, Union
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from enum import Enum

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional

from allophant import utils
from allophant.dataset_processing import Batch
from allophant.config import (
    Glu1dConfig,
    DropoutConfig,
    LayerNormConfig,
    MaxPoolingConfig,
    FrontendConfig,
    DirectFrontendConfig,
    LinearFrontendConfig,
    SequentialFrontendConfig,
)
from allophant.network.padding import VariableLengthReflectPad
from allophant.network import padding


T = TypeVar("T", bound=Union[float, Tensor])


class ShapeMode(Enum):
    LENGTH_FIRST = 0
    BATCH_FIRST = 1


@dataclass
class MaskInfo:
    pattern: Sequence[int] = (-1, -1)
    shape_mode: ShapeMode = ShapeMode.BATCH_FIRST

    def __post_init__(self):
        self._mask_pattern = tuple(slice(None) if dim == -1 else None for dim in self.pattern)

    def to_shape(self, mask: Tensor) -> Tensor:
        if self.shape_mode == ShapeMode.LENGTH_FIRST:
            mask = mask.T
        return mask[self._mask_pattern]


class LengthWrapper(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        length_function: Optional[Callable[[Tensor], Tensor]] = None,
        mask_info: Optional[MaskInfo] = None,
        pass_length_info: bool = False,
    ):
        super().__init__()
        self._length_function = length_function
        self._mask_input = mask_info
        self._pass_length_info = pass_length_info
        self.module = module

    def forward(self, inputs: Tensor, lengths: Tensor) -> Tuple[Tensor, Tensor]:
        if self._length_function is None:
            if self._pass_length_info:
                return self.module(inputs, lengths), lengths
            return self.module(inputs), lengths

        if self._mask_input is not None:
            inputs *= self._mask_input.to_shape(utils.mask_sequence(lengths))
        out_lengths = self._length_function(lengths)
        if self._pass_length_info:
            return self.module(inputs, lengths), out_lengths
        return self.module(inputs), out_lengths

    def lengths(self, lengths: Tensor) -> Tensor:
        if self._length_function is None:
            return lengths
        return self._length_function(lengths)


class LengthSequential(nn.Module):
    def __init__(self, *args: LengthWrapper):
        super().__init__()
        self.layers = nn.ModuleList(args)

    def forward(self, inputs: Tensor, lengths: Tensor) -> Tuple[Tensor, Tensor]:
        for layer in self.layers:
            inputs, lengths = layer(inputs, lengths)
        return inputs, lengths

    def lengths(self, lengths: Tensor) -> Tensor:
        for layer in self.layers:
            lengths = layer.lengths(lengths)  # type: ignore
        return lengths


class Glu1d(nn.Module):
    def __init__(
        self,
        input_dimensions: int,
        output_dimensions: int,
        kernel_size: int,
        stride: int = 1,
        reflect_pad: bool = True,
    ):
        super().__init__()
        self._padding = padding.get_padding(kernel_size, stride)
        self._reflect_padding = VariableLengthReflectPad(self._padding) if reflect_pad else None
        self._kernel_size = kernel_size
        self._stride = stride
        self._weights = nn.Conv1d(
            input_dimensions,
            output_dimensions * 2,
            kernel_size=kernel_size,
            stride=stride,
        )

    @property
    def padding(self) -> Tuple[int, int]:
        return self._padding

    @property
    def kernel_size(self) -> int:
        return self._kernel_size

    @property
    def stride(self) -> int:
        return self._stride

    def forward(self, inputs: Tensor, lengths: Tensor) -> Tensor:
        if self._reflect_padding is None:
            inputs = functional.pad(inputs, self._padding)
        else:
            inputs = self._reflect_padding(inputs, lengths)
        return functional.glu(self._weights(inputs), 1)


class Frontend(nn.Module, metaclass=ABCMeta):
    _output_dimensions: int

    @abstractmethod
    def forward(self, batch: Batch) -> Batch:
        pass

    @property
    def output_dimensions(self) -> int:
        return self._output_dimensions

    def lengths(self, input_lengths: Tensor) -> Tensor:
        return input_lengths


class DirectFrontend(Frontend):
    def __init__(self, config: DirectFrontendConfig, feature_size: int):
        super().__init__()
        self._output_dimensions = feature_size
        self._dropout = nn.Dropout(config.input_dropout) if config.input_dropout > 0 else None

    def forward(self, batch: Batch) -> Batch:
        output = batch.audio_features
        if self._dropout is not None:
            output = self._dropout(output)
        return Batch(output, batch.lengths, batch.language_ids)


class LinearFrontend(Frontend):
    def __init__(self, config: LinearFrontendConfig, feature_size: int, elementwise_affine: bool = False):
        super().__init__()
        self._output_dimensions = config.neurons
        linear = nn.Linear(feature_size, config.neurons)
        if config.input_dropout > 0:
            self._layer = nn.Sequential(
                nn.Dropout(config.input_dropout),
                nn.LayerNorm(feature_size, elementwise_affine=elementwise_affine),
                linear,
                nn.LeakyReLU(),
            )
        else:
            self._layer = nn.Sequential(
                nn.LayerNorm(feature_size, elementwise_affine=elementwise_affine),
                linear,
                nn.LeakyReLU(),
            )

    def forward(self, batch: Batch) -> Batch:
        return Batch(
            self._layer(batch.audio_features.transpose(1, 2)).transpose(1, 2), batch.lengths, batch.language_ids
        )


def conv_length(
    kernel_size: int, stride: int = 1, use_padding: bool = True, stft_type: bool = False
) -> Callable[[Tensor], Tensor]:
    if use_padding:
        layer_padding = sum(padding.get_padding(kernel_size, stride, stft_type))
    else:
        layer_padding = 0

    def padded_length(lengths: Tensor) -> Tensor:
        return torch.div(((lengths + layer_padding) - kernel_size), stride, rounding_mode="floor") + 1

    return padded_length


class Transpose(nn.Module):
    def __init__(self, dimension_a: int, dimension_b: int):
        super().__init__()
        self._dimension_a = dimension_a
        self._dimension_b = dimension_b

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs.transpose(self._dimension_a, self._dimension_b)


SequentialFrontendCls = TypeVar("SequentialFrontendCls", bound="SequentialFrontend")


class SequentialFrontend(Frontend):
    def __init__(self, layers: LengthSequential, output_dimensions: int, upscale_factor: float = 1):
        super().__init__()
        self._layers = layers
        self._output_dimensions = output_dimensions
        self._upscale_factor = upscale_factor

    @classmethod
    def from_config(
        cls: Type[SequentialFrontendCls], config: SequentialFrontendConfig, feature_size: int
    ) -> SequentialFrontendCls:
        layers = []
        previous_output_size = feature_size
        upscale_factor = 1
        for layer in config.layers:
            if isinstance(layer, DropoutConfig):
                layers.append(LengthWrapper(nn.Dropout(layer.rate)))
            elif isinstance(layer, Glu1dConfig):
                output_size = layer.out_channels
                module = Glu1d(previous_output_size, output_size, layer.kernel, layer.stride)
                layers.append(
                    LengthWrapper(
                        module,
                        conv_length(module.kernel_size, module.stride),
                        MaskInfo((-1, 1, -1)),
                        pass_length_info=True,
                    )
                )
                previous_output_size = output_size
                upscale_factor *= module.stride
            elif isinstance(layer, LayerNormConfig):
                layers.append(
                    LengthWrapper(
                        nn.Sequential(
                            Transpose(-1, -2),
                            nn.LayerNorm(previous_output_size, elementwise_affine=layer.affine),
                            Transpose(-2, -1),
                        )
                    )
                )
            elif isinstance(layer, MaxPoolingConfig):
                layers.append(LengthWrapper(nn.MaxPool1d(layer.size), conv_length(layer.size), MaskInfo((-1, 1, -1))))
                upscale_factor *= layer.size
            else:
                raise ValueError(f"Unsupported layer config of type: {layer.__class__.__name__}")

        return cls(LengthSequential(*layers), previous_output_size, upscale_factor)

    @property
    def upscale_factor(self) -> float:
        return self._upscale_factor

    def forward(self, batch: Batch) -> Batch:
        features, lengths = self._layers(batch.audio_features, batch.lengths)
        return Batch(features, lengths, batch.language_ids)

    def downsampled_lengths(self, lengths: Tensor) -> Tensor:
        return self._layers.lengths(lengths)


def frontend_from_config(
    frontend_config: FrontendConfig, feature_size: int, elementwise_affine: bool = False
) -> Frontend:
    if isinstance(frontend_config, DirectFrontendConfig):
        return DirectFrontend(frontend_config, feature_size)
    if isinstance(frontend_config, LinearFrontendConfig):
        return LinearFrontend(frontend_config, feature_size, elementwise_affine)

    raise ValueError(f"Unsupported frontend config type {frontend_config.__class__.__name__}")
