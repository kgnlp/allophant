from collections.abc import Callable
import typing
from typing import Tuple

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional
from torchaudio.transforms import MFCC, MelSpectrogram

from allophant.utils import Squeeze, Unsqueeze, LogCompression
from allophant.config import Config, FeatureType, Window
from allophant.network import padding


class PreEmphasis(nn.Module):
    def __init__(self, coefficient: float = 0.97, trainable: bool = False):
        super().__init__()
        self.coefficient = coefficient
        # Uses symmetric 1 padding though only initial padding is relevant
        self._pre_emphasis_convolution = nn.Conv1d(1, 1, kernel_size=2, bias=False)
        with torch.no_grad():
            # Initializes to [-0.97, 1] which is equivalent to the regular pre-emphasis formula
            self._pre_emphasis_convolution.weight.copy_(torch.tensor([-coefficient, 1]))
        if not trainable:
            self._pre_emphasis_convolution.requires_grad_(False)
        self._padding = padding.get_padding(2, stft_type=True)

    def forward(self, batch: Tensor) -> Tensor:
        if batch.ndim == 3:
            return self._pre_emphasis_convolution(functional.pad(batch, self._padding, "reflect"))
        # Reflect padding to match lengths of in/out
        return self._pre_emphasis_convolution(functional.pad(batch.view(1, 1, -1), self._padding, "reflect")).ravel()


def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def frame_count(num_samples: int, stride_samples: int) -> int:
    # Chosen to be equivalent to the torch.sfft implementation
    return 1 + (num_samples // stride_samples)


def _mono_channel(audio: Tensor) -> Tensor:
    # Assumes channel first
    return audio[0]


class FeatureFunction:
    def __init__(
        self,
        name: str,
        function: Callable[[Tensor], Tensor],
        sample_rate: int,
        dimensions: Tuple[int | None, int] | Tuple[int, int | None] | None = None,
        window: Window | None = None,
    ):
        self._name = name
        self._function = function
        self._stride_samples = window.frame_stride_samples(sample_rate) if window is not None else None
        # Transpose if lengths are specified before features
        has_dimensions = dimensions is not None
        self._transpose_required = False
        if has_dimensions:
            # Transpose dimensions if necessary
            if dimensions[1] is None:
                self._transpose_required = True
                dimensions = (None, dimensions[0])
            self._max_dimensions = typing.cast(Tuple[None, int], dimensions)
            self._feature_size = self._max_dimensions[1]
            self._dimensions = (1, self._feature_size)
        else:
            self._max_dimensions = (None,)
            self._feature_size = 1
            self._dimensions = (1,)

    @property
    def name(self) -> str:
        return self._name

    @property
    def feature_size(self) -> int:
        return self._feature_size

    @property
    def dimensions(self) -> Tuple[int, int] | Tuple[int]:
        return self._dimensions

    def frame_count(self, num_samples: int) -> int:
        if not self._stride_samples:
            return num_samples
        return frame_count(num_samples, self._stride_samples)

    def __call__(self, waveform: Tensor) -> Tensor:
        processed = self._function(waveform)
        if self._transpose_required:
            return processed.transpose(-2, -1)
        return processed

    @classmethod
    def from_config(cls, config: Config, sample_rate: int):
        feature_type = config.preprocessing.feature_type
        num_filters = config.preprocessing.num_filters
        n_fft = next_power_of_2(int((1024 / 16_000) * sample_rate))

        if feature_type == FeatureType.MFCC:
            window = config.preprocessing.window
            layers = [
                PreEmphasis(),
                MFCC(
                    sample_rate,
                    n_mfcc=num_filters,
                    log_mels=False,
                    melkwargs={
                        "n_fft": n_fft,
                        "win_length": window.frame_duration_samples(sample_rate),
                        "hop_length": window.frame_stride_samples(sample_rate),
                    },
                ),
                Unsqueeze(0),
                # CMVN
                nn.InstanceNorm1d(num_filters),
                Squeeze(0),
            ]
            return cls(
                feature_type.value,
                nn.Sequential(*layers),
                sample_rate,
                (num_filters, None),
                window,
            )
        elif feature_type == FeatureType.FILTERBANKS:
            window = config.preprocessing.window
            layers = [
                PreEmphasis(),
                MelSpectrogram(
                    sample_rate,
                    n_fft=n_fft,
                    win_length=window.frame_duration_samples(sample_rate),
                    hop_length=window.frame_stride_samples(sample_rate),
                    n_mels=num_filters,
                ),
                LogCompression(),
                Unsqueeze(0),
                # CMVN
                nn.InstanceNorm1d(num_filters),
                Squeeze(0),
            ]
            return cls(
                feature_type.value,
                nn.Sequential(*layers),
                sample_rate,
                (num_filters, None),
                window,
            )
        elif feature_type == FeatureType.RAW:
            # Just raw audio without a filter
            return cls("raw", _mono_channel, sample_rate)

        raise ValueError(f"Unsupported feature type: {feature_type}")
