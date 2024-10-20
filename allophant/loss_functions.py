from torch import Tensor
from torch import nn
from torch.nn import functional
from abc import ABCMeta, abstractmethod


class LossWrapper(nn.Module, metaclass=ABCMeta):
    _loss: nn.Module

    @property
    def wrapped_loss(self):
        return self._loss

    @abstractmethod
    def forward(self, logits: Tensor, labels: Tensor, predicted_lengths: Tensor, label_lengths: Tensor) -> Tensor:
        pass


class CTCWrapper(LossWrapper):
    def __init__(self):
        super().__init__()
        # Enables zero losses if the label sequence is longer than the input
        # sequence to be robust against outliers
        self._loss = nn.CTCLoss(reduction="sum", zero_infinity=True)

    def forward(self, logits: Tensor, labels: Tensor, predicted_lengths: Tensor, label_lengths: Tensor) -> Tensor:
        return self._loss(functional.log_softmax(logits, -1), labels, predicted_lengths, label_lengths)


class SequenceCrossEntropyWrapper(LossWrapper):
    def __init__(self, label_smoothing: float = 0):
        super().__init__()
        # Enables zero losses if the label sequence is longer than the input
        # sequence to be robust against outliers
        self._loss = nn.CrossEntropyLoss(reduction="sum", label_smoothing=label_smoothing)

    def forward(
        self,
        logits: Tensor,
        labels: Tensor,
        _predicted_lengths: Tensor | None = None,
        _label_lengths: Tensor | None = None,
    ) -> Tensor:
        return self._loss(logits, labels)
