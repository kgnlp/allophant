from abc import ABCMeta
import typing
from typing import (
    ClassVar,
    Type,
    TypeVar,
    Iterable,
    Dict,
    Protocol,
    Optional,
    Any,
    Sequence,
    Mapping,
    Generic,
    List,
    Union,
    Tuple,
    runtime_checkable,
)
import dataclasses
from dataclasses import dataclass
from enum import Enum
import re
from re import Pattern

import toml
from marshmallow import Schema
from marshmallow.exceptions import ValidationError
from marshmallow.fields import Nested, Field, String
from marshmallow.validate import OneOf, Range
from marshmallow import fields
from marshmallow_oneofschema import OneOfSchema
from marshmallow_enum import EnumField
import marshmallow_dataclass
from torch import optim
from torch.optim import Optimizer as TorchOptimizer
from torch import nn

from allophant.loss_functions import LossWrapper, CTCWrapper, SequenceCrossEntropyWrapper
from allophant import utils


F = TypeVar("F")


@runtime_checkable
class KeyedClass(Protocol):
    """Protocol for classes that have a TYPE class variable indicating its key in configs"""

    TYPE: ClassVar[str]


class KeyedOneOfSchema(OneOfSchema):
    """
    A Schema for (de)serializing a union of types tagged with a key field
    """

    types: ClassVar[Tuple[Type[KeyedClass], ...]]

    def get_obj_type(self, obj: object) -> str:
        if isinstance(obj, self.types):
            return obj.TYPE
        raise ValidationError(f"Unknown object type: {obj!r}")


def _choices(keyed_schemas: Iterable[Type[KeyedClass]]) -> Dict[str, Type[Schema]]:
    return {schema.TYPE: marshmallow_dataclass.class_schema(schema) for schema in keyed_schemas}


def _nested_field(schema: Type[Schema], field_args: Dict[str, Any] | None = None, **kwargs) -> Any:
    return utils.schema_field(Nested(schema, **(field_args if field_args is not None else {})), **kwargs)


def _nested_list_field(schema: Type[Schema], field_args: Dict[str, Any] | None = None, **kwargs) -> Any:
    return utils.schema_field(fields.List(Nested(schema, **(field_args if field_args is not None else {}))), **kwargs)


@dataclass
class WarmupInfo:
    """
    Configures learning rate warmup for transformers as derived
    from https://nlp.seas.harvard.edu/2018/04/03/attention.html#optimizer

    :param model_size: The input and output size of transformer blocks used in
        warmup calculations (`d_model`)
    """

    model_size: int


@marshmallow_dataclass.add_schema
@dataclass
class _WarmupState:
    """
    Contains the current step and learning rate for learning rate warmup

    :param step: current warmup step
    :param rate: current learning rate
    """

    Schema: ClassVar[Type[Schema]]

    step: int = 0
    rate: float = 0


class WarmupScheduler:
    _warmup_state: _WarmupState

    def __init__(
        self,
        optimizer: TorchOptimizer,
        warmup_info: WarmupInfo,
        warmup_steps: int,
        constant_steps: int = 0,
        factor: int = 2,
    ) -> None:
        self._optimizer = optimizer

        self._warmup_steps = warmup_steps
        self._constant_steps = constant_steps
        self._steps_until_decay = warmup_steps + constant_steps
        self._factor = factor
        self._model_size = warmup_info.model_size

        self._warmup_state = _WarmupState(1, 0)
        self._warmup_state.rate = self._rate(1)
        # Set initial learning rate
        self._set_lr(self._warmup_state.rate)

    def _set_lr(self, rate: float) -> None:
        for parameter in self._optimizer.param_groups:
            parameter["lr"] = rate

    @property
    def last_lr(self) -> float:
        return self._warmup_state.rate

    def _rate(self, step=None) -> float:
        """
        Computes the learning rate for the given warmup step
        as derived from https://nlp.seas.harvard.edu/2018/04/03/attention.html#optimizer

        :param step: Current step or `None` for the initial step

        :return: The new learning rate for the given step
        """
        # Based on implementation: https://nlp.seas.harvard.edu/2018/04/03/attention.html#optimizer
        if step is None:
            step = self._warmup_state.step

        # Warmup
        if step < self._warmup_steps:
            return self._factor * (self._model_size ** (-0.5) * (step * self._warmup_steps ** (-1.5)))
        # Keep constant at the maximum learning rate
        if step < self._steps_until_decay:
            return self._factor * (self._model_size ** (-0.5) * (self._warmup_steps ** (-0.5)))
        # Decay
        else:
            return self._factor * (self._model_size ** (-0.5) * ((step - self._constant_steps) ** (-0.5)))

    def step(self) -> None:
        # Based on implementation: https://nlp.seas.harvard.edu/2018/04/03/attention.html#optimizer
        self._warmup_state.step += 1
        rate = self._rate()
        self._set_lr(rate)
        self._warmup_state.rate = rate

    def state_dict(self) -> Dict[str, Any]:
        return {"warmup_state": _WarmupState.Schema().dump(self._warmup_state)}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self._warmup_state = _WarmupState.Schema().load(state_dict["warmup_state"])  # type: ignore


@dataclass
class WarmupConfig(KeyedClass):
    """
    Contains learning rate warmup parameters for transformers as derived
    from https://nlp.seas.harvard.edu/2018/04/03/attention.html#optimizer
    but with the addition of keeping the learning rate constant for a while until decaying

    :param warmup_steps: Number of batch updates until warmup completes
    :param constant_steps: The number of steps to keep the learning rate
        constant until decay. The schedule is equivalent to the standard
        transformer schedule if this is kept at 0
    :param factor: Scaling factor for warmup updates
    """

    TYPE: ClassVar[str] = "warmup"

    warmup_steps: int
    constant_steps: int = 0
    factor: int = 2

    def make_scheduler(self, optimizer: TorchOptimizer, warmup_info: WarmupInfo) -> WarmupScheduler:
        return WarmupScheduler(optimizer, warmup_info, self.warmup_steps, self.constant_steps, self.factor)


class LrSchedulerSchema(KeyedOneOfSchema):
    """Schema which selects between learning rate schedules"""

    type_field = "type"
    types = (WarmupConfig,)
    type_schemas = _choices(types)


# Lr schedulers that are available
LrSchedulerConfig = WarmupConfig


@dataclass
class OptimizerWrapper:
    """
    Wrapper around a `torch.Optimizer` with optional learning rate decay
    """

    def __init__(self, optimizer: TorchOptimizer, warmup_info: WarmupInfo):
        """
        Initializes a new optimizer wrapper with an optimizer and optional learning rate decay

        :param optimizer: A `torch.Optimizer`
        :param warmup_info: Information about the model required for configuring Warmup schedulers
        """
        self._optimizer = optimizer
        self._warmup_info = warmup_info

    def add_schedulers(self, scheduler_config: LrSchedulerConfig | None) -> None:
        """
        Adds a Learning Rate schedulers generated from the provided config

        :param schedule_configs: Configuration of a learning rate scheduler
        """
        self._lr_scheduler = (
            None if scheduler_config is None else scheduler_config.make_scheduler(self._optimizer, self._warmup_info)
        )

    @property
    def optimizer(self) -> TorchOptimizer:
        return self._optimizer

    def step(self) -> None:
        """
        Performs an optimization step and steps the learning rate scheduler if one exists
        """
        self.optimizer.step()
        if self._lr_scheduler is not None:
            self._lr_scheduler.step()

    @property
    def param_groups(self) -> List[Dict[Any, Any]]:
        return self._optimizer.param_groups

    def current_learning_rate(self) -> float:
        return self.param_groups[0]["lr"]

    def state_dict(self) -> Dict[str, Any]:
        """
        Returns the state of the OptimizerWrapper as a dict

        :return: The warmup, optimizer and scheduler states depending on configuration
        """
        return {
            "lr_scheduler": None if self._lr_scheduler is None else self._lr_scheduler.state_dict(),
            "optimizer": self._optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """
        Loads the optimizer wrapper state

        :param state_dict: The states of warmup, the optimizer and the schedulers depending on configuration.
            Should be a `dict` returned from a call to `state_dict`
        """
        self._optimizer.load_state_dict(state_dict["optimizer"])
        if self._lr_scheduler is not None:
            self._lr_scheduler.load_state_dict(state_dict["lr_scheduler"])


@dataclass
class Optimizer(metaclass=ABCMeta):
    """Superclass for all optimizer configurations which support learning rates and weight decay"""

    learning_rate: float
    l2_regularization: float = 0

    def get_optimizer(self, _params: Iterable[nn.parameter.Parameter], _warmup_info: WarmupInfo) -> OptimizerWrapper:
        """
        Retrieves an `OptimizerWrapper` based on the configuration options

        :param params: Parameters to be optimized
        :param warmup_info: Configuration and parameter information for learning rate warmup

        :return: An `OptimizerWrapper` optimizing `params`, optionall with learning rate warmup
        """
        raise NotImplementedError("No torch optimizer defined for this class")


@dataclass
class SGD(Optimizer, KeyedClass):
    """
    Configuration for the stochastic gradient descent (SGD) optimizer
    supporting weight decay, momentum and learning rate decay
    """

    TYPE: ClassVar[str] = "sgd"

    momentum: float = 0

    def get_optimizer(self, params: Iterable[nn.parameter.Parameter], warmup_info: WarmupInfo) -> OptimizerWrapper:
        sgd = optim.SGD(params, self.learning_rate, self.momentum, weight_decay=self.l2_regularization)
        return OptimizerWrapper(sgd, warmup_info)


@dataclass
class Adam(Optimizer, KeyedClass):
    """
    Configuration for the ADAM optimizer supporting weight and learning rate decay
    """

    TYPE: ClassVar[str] = "adam"

    learning_rate: float = 0.01
    beta_1: float = 0.9
    beta_2: float = 0.98

    def get_optimizer(self, params: Iterable[nn.parameter.Parameter], warmup_info: WarmupInfo) -> OptimizerWrapper:
        adam = optim.Adam(
            params, self.learning_rate, betas=(self.beta_1, self.beta_2), weight_decay=self.l2_regularization
        )
        return OptimizerWrapper(
            # Same betas as in post-lm and attention is all you need
            adam,
            warmup_info,
        )


class OptimizerSchema(KeyedOneOfSchema):
    """Schema which selects between optimizers"""

    type_field = "algorithm"
    types = (SGD, Adam)
    type_schemas = _choices(types)


@dataclass
class Window:
    """
    Window configuration for acoustic features operating on frames such as
    STFT-based features like Mel Filterbanks or MFCCs

    :param frame_duration: Duration of a frame in milliseconds
    :param frame_stride: Stride of the filter in milliseconds - also known as "hop"
    """

    frame_duration: int
    frame_stride: int

    @staticmethod
    def _ms_to_samples(milliseconds: int, sample_rate: int) -> int:
        """
        Converts milliseconds to the number of samples at the given sample rate

        :param milliseconds: Time in milliseconds to convert to samples
        :param sample_rate: Sample rate used in the conversion

        :return: Number of samples for the given duration at the given sample rate
        """
        return int((milliseconds / 1000) * sample_rate)

    def frame_duration_samples(self, sample_rate: int) -> int:
        """
        Calculates the frame duration in samples at the given sample rate

        :param sample_rate: Sample rate used for the conversion

        :return: The frame duration in samples at the given sample rate
        """
        return self._ms_to_samples(self.frame_duration, sample_rate)

    def frame_stride_samples(self, sample_rate: int) -> int:
        """
        Calculates the frame stride in samples at the given sample rate

        :param sample_rate: Sample rate used for the conversion

        :return: The frame stride in samples at the given sample rate
        """
        return self._ms_to_samples(self.frame_stride, sample_rate)


@dataclass
class DropoutConfig(KeyedClass):
    """
    Defines a dropout regularization layer

    :param rate: Probability of neurons in the input being replaced with zeros ("dropped out")
    """

    TYPE: ClassVar[str] = "dropout"

    rate: float = 0


@dataclass
class LayerNormConfig(KeyedClass):
    """
    Defines a layer normalization layer

    :param affine: If true, an affine transformation with element-wise scale and bias parameters is included
    """

    TYPE: ClassVar[str] = "layer_norm"

    affine: bool = False


@dataclass
class Glu1dConfig(KeyedClass):
    """
    Defines a one-dimensional Gated Linear Unit (GLU) layer over time

    :param out_channels: Number of filters (output channels) for the given kernel
    :param kernel: Width of the kernel
    :param stride: Stride of the kernel
    """

    TYPE: ClassVar[str] = "glu1d"

    out_channels: int
    kernel: int
    stride: int = 1


@dataclass
class MaxPoolingConfig(KeyedClass):
    """
    Defines a one-dimensional max pooling layer with a stride of one

    :param size: Size of the pooling operation
    """

    TYPE: ClassVar[str] = "max_pool"

    size: int


@dataclass
class TransformerConfig(KeyedClass):
    """
    Configures a Pre-LN transformer

    :param feedforward_neurons: Number of neurons in the feedforward sub-block
    :param heads: Number of heads in the multi-head attention sub-block
    :param activation: Activation function used between the multi-head
        attention and feedforward sub-block - either "relu" or "gelu"
    :param num_layers: Number of transformer blocks
    :param dropout_rate: Dropout rate used throughout the transformer for regularization
    :param positional_embeddings: If true, sinusoidal positional embeddings are
        added to the inputs before passing them to the transformer
    """

    TYPE: ClassVar[str] = "transformer"

    feedforward_neurons: int
    heads: int
    activation: str = typing.cast(
        str, utils.schema_field(String(default="relu", missing="relu", validate=OneOf({"relu", "gelu"})))
    )
    num_layers: int = 1
    dropout_rate: float = 0
    positional_embeddings: bool = True


class LayerSchema(KeyedOneOfSchema):
    """Schema which selects between optimizers"""

    type_field = "type"
    types = (Glu1dConfig, MaxPoolingConfig, DropoutConfig, LayerNormConfig)
    type_schemas = _choices(types)


Layer = Union[Glu1dConfig, MaxPoolingConfig, DropoutConfig, LayerNormConfig]


@dataclass
class DirectFrontendConfig(KeyedClass):
    """
    Defines a direct frontend which passes input features through as-is and optionally applies input dropout

    :param input_dropout: Amount of dropout added to the input features
    """

    TYPE: ClassVar[str] = "direct"

    input_dropout: float = 0


@dataclass
class LinearFrontendConfig(KeyedClass):
    """
    Defines a frontend with a single affine transformation and optional input dropout
    which allows the feature dimensions to be "upscaled" before passing the input to the transformer

    :param neurons: Number of neurons in the affine transformation, defines the
        input size to the transformer (`d_model`)
    :param input_dropout: Amount of dropout added to the input features
    """

    TYPE: ClassVar[str] = "linear"

    neurons: int
    input_dropout: float = 0


@dataclass
class SequentialFrontendConfig:
    """
    Defines an arbitrarily complex sequential config from a list of supported layer configurations

    :param layers: A list of supported layer configurations
    """

    layers: List[Layer] = _nested_list_field(LayerSchema)


class FrontendSchema(KeyedOneOfSchema):
    """Schema which selects between frontend architectures"""

    type_field = "architecture"
    types = (DirectFrontendConfig, LinearFrontendConfig)
    type_schemas = _choices(types)


class LossConfig(metaclass=ABCMeta):
    """
    Configures a loss function for training and validation
    """

    BLANK_OFFSET: ClassVar[int] = 0

    def get_loss(self) -> LossWrapper:
        """Retrieves the loss wrapper for a loss configuration"""
        raise NotImplementedError("No loss function defined for this class")


@dataclass
class CTCLossConfig(KeyedClass, LossConfig):
    """
    Represents regular CTC loss
    """

    TYPE: ClassVar[str] = "CTC"
    # Offset for CTC blank label
    BLANK_OFFSET: ClassVar[int] = 1

    def get_loss(self) -> CTCWrapper:
        return CTCWrapper()


@dataclass
class SequenceCrossEntropyLossConfig(KeyedClass, LossConfig):
    """
    Represents cross-entropy loss which takes the mean pooling output of the acoustic model
    instead of being applied frame-level

    :param label_smoothing: Amount of smoothing during loss computation in [0, 1], where zero disables smoothing
    """

    TYPE: ClassVar[str] = "sequence-cross-entropy"

    label_smoothing: float = dataclasses.field(default=0, metadata={"validate": Range(min=0, max=1)})

    def get_loss(self) -> SequenceCrossEntropyWrapper:
        return SequenceCrossEntropyWrapper(self.label_smoothing)


class MainLossSchema(KeyedOneOfSchema):
    """Schema which selects between loss types"""

    type_field = "type"
    types = (CTCLossConfig,)
    type_schemas = _choices(types)


class ClassifierLossSchema(KeyedOneOfSchema):
    """Schema which selects between loss types"""

    type_field = "type"
    types = (CTCLossConfig, SequenceCrossEntropyLossConfig)
    type_schemas = _choices(types)


ClassifierLossConfig = CTCLossConfig | SequenceCrossEntropyLossConfig


@dataclass
class MultiheadAttentionConfig:
    """
    A simple Multi Head Attention layer

    :param num_heads: The number of attention heads
    :param positional_embeddings: Whether to add sinusoidal positional
        embeddings to the projected attention layer input before applying multi-head attention
    """

    TYPE: ClassVar[str] = "multi-head-attention"

    num_heads: int = 1
    positional_embeddings: bool = False


class TimeLayerSchema(KeyedOneOfSchema):
    """
    Schema which selects between layer types that encode information over
    time for classifier layers in the hierarchical projection
    """

    type_field = "type"
    types = (MultiheadAttentionConfig,)
    type_schemas = _choices(types)


@dataclass
class ProjectionEntryConfig:
    """
    Configuration for a single classifier node in the hierarchical classifier and its dependencies.
    Each classifier needs at least one dependency as an input

    :param name: Name of the classifier. Should correspond to feature names
    :param dependencies: A sequence of classifier names or "OUTUT" which specifies the output of acoustic model.
        Additionally, specific layers in the acoustic model can be used as dependencies such as "OUTPUT_0" for the first hidden layer.
    :param loss: The loss applied to the classifier when training. Defaults to CTC loss
    """

    OUTPUT_DEPENDENCY: ClassVar[str] = "OUTPUT"
    OUTPUT_PATTERN: ClassVar[Pattern] = re.compile(rf"^{OUTPUT_DEPENDENCY}(?:_(\d+))?$")
    PHONEME_LAYER: ClassVar[str] = "phoneme"
    PHONE: ClassVar[str] = "phone"

    name: str
    dependencies: List[str] = dataclasses.field(default_factory=lambda: [ProjectionEntryConfig.OUTPUT_DEPENDENCY])
    time_layer: Optional[MultiheadAttentionConfig] = _nested_field(TimeLayerSchema, {"allow_none": True}, default=None)
    loss: ClassifierLossConfig = _nested_field(ClassifierLossSchema, default_factory=CTCLossConfig)


class FeatureSet(Enum):
    """
    The phonetic feature set to use
    """

    PHOIBLE = "phoible"
    PANPHON = "panphon"


class PhonemeLayerType(Enum):
    """
    Represents the type of layer used for phoneme classification
    """

    SHARED = "shared"
    PRIVATE = "private"
    ALLOPHONES = "allophones"


@dataclass
class EmbeddingCompositionConfig:
    """
    Configures an layer that uses bags of embeddings instead of linear
    projection layers for phones (with the allophone layer) and phonemes (with a shared phoneme model)

    :param embedding_size: The embedding dimension of each attribute embedding
        and, as a result, each phone or phoneme embedding vector
    """

    embedding_size: int


@dataclass
class ProjectionConfig:
    """
    Configures a hierarchical projection layer

    :param classes: A sequence of named classifiers with dependencies
    :param feature_set: The phonetic feature set for which attribute classifiers can be defined
    :param phoneme_layer: Selects the type of layer used for phoneme classification
    :param acoustic_model_dropout: Dropout to apply to (intermediate) output
        layers of the transformer before they are used as dependencies of the classifier layers
    :param dependency_blanks: If false, logits for the CTC "BLANK" class are
        removed before passing class logits through a softmax function and using
        them as the input of another classifier. Note that this only applies
        when the classifier output is used as a dependency and doesn't affect the per-class probabilities that
        can be directly accessed during training, validation or prediction
    :param allophone_l2_alpha: The alpha parameter for the allophone matrix l2
        penalty. Only used when `phoneme_layer` is set to
        :py:const:`PhonemeLayerType.PRIVATE` or :py:const:`PhonemeLayerType.ALLOPHONES`
    :param embedding_composition: Uses bags of embeddings instead of linear projection layers for phones (with the allophone layer) and phonemes (with a shared phoneme model)
    """

    classes: List[ProjectionEntryConfig]
    feature_set: FeatureSet = utils.schema_field(EnumField(FeatureSet, by_value=True), default=FeatureSet.PHOIBLE)
    phoneme_layer: PhonemeLayerType = utils.schema_field(
        EnumField(PhonemeLayerType, by_value=True), default=PhonemeLayerType.SHARED
    )
    acoustic_model_dropout: float = 0
    dependency_blanks: bool = True
    # Default value is the same as in "Universal Phone Recognition With a Multilingual Allophone Systems" by Li et al.
    allophone_l2_alpha: float = 10
    embedding_composition: Optional[EmbeddingCompositionConfig] = None

    def loss_functions(self) -> Dict[str, LossWrapper]:
        return {classifier.name: classifier.loss.get_loss() for classifier in self.classes}


FrontendConfig = DirectFrontendConfig | LinearFrontendConfig


@dataclass
class TransformerAcousticModelConfig(KeyedClass):
    """
    Defines a transformer acoustic model including architecture specific frontends

    :param transformer: Configuration for the pre-LN transformer
    :param frontend: Configuration of the frontend architecture
    :param sequential_frontend: Configuration for sequential frontend layers following the `frontend`
    :param elementwise_affine: If true, an affine transformation with
        element-wise scale and bias parameters is included in layer normalization
        immediately before, after and in the transformer acoustic model
    """

    TYPE: ClassVar[str] = "pre-ln-transformer"

    transformer: TransformerConfig
    frontend: FrontendConfig = _nested_field(FrontendSchema)
    sequential_frontend: Optional[SequentialFrontendConfig] = None
    elementwise_affine: bool = False


@dataclass
class Wav2Vec2Config(KeyedClass):
    TYPE: ClassVar[str] = "wav2vec2"


@dataclass
class UnfreezeScheduleConfig:
    """
    Declares the number of updates before unfreezing a Wav2Vec2 submodule.
    If a submodule is already unfrozen, the scheduled steps have no effect

    :param feature_encoder_steps: Nmber of steps before unfreezing the feature encoder
    :param feature_projection_steps: Nmber of steps before unfreezing the feature projection layer after the feature encoder
    :param encoder_steps: Nmber of steps before unfreezing the encoder layer after the feature projection
    """

    feature_encoder_steps: Optional[int] = None
    feature_projection_steps: Optional[int] = None
    encoder_steps: Optional[int] = None


@dataclass
class Wav2Vec2PretrainedConfig(KeyedClass):
    """
    Declares a Wav2Vec2 pre-trained model from `huggingface <https://huggingface.co/models?other=wav2vec2>`_

    :param model_id: ID of a huggingface repository containing a Wav2Vec2 pre-trained model
    :param freeze_feature_encoder: Freezes the feature encoder
    :param freeze_feature_projection: Freezes the feature projection layer after the feature encoder
    :param freeze_encoder: Freezes the encoder layer after the feature projection
    :param unfreeze_schedule: Allows unfreezing Wav2Vec2 submodules after a number of updates
    """

    TYPE: ClassVar[str] = "wav2vec2-pretrained"

    model_id: str
    freeze_feature_encoder: bool = True
    freeze_feature_projection: bool = False
    freeze_encoder: bool = False
    unfreeze_schedule: Optional[UnfreezeScheduleConfig] = None


AcousticModel = TransformerAcousticModelConfig | Wav2Vec2Config | Wav2Vec2PretrainedConfig


class AcousticModelSchema(KeyedOneOfSchema):
    """Schema which selects between acoustic models"""

    type_field = "type"
    types = (TransformerAcousticModelConfig, Wav2Vec2Config, Wav2Vec2PretrainedConfig)
    type_schemas = _choices(types)


class BatchingMode(Enum):
    """
    Represents the type of batching used which changes the unit represented by
    the provided batch size.

    With `BatchingMode.FRAMES` the `batch_size` represents the maximum number
    of frames per batch.
    With `BatchingMode.UTTERANCES` the `batch_size` represents the maximum
    number of utterances per batch. This corresponds to regular batch size.
    """

    FRAMES = "frames"
    UTTERANCES = "utterances"


@dataclass
class Architecture:
    """
    Configuration options for the acoustic model architecture

    :param batch_size: The batch size used during training. Its unit depends on the value of `batching_mode`
    :param projection: Configuration of the hierarchical projection layer
    :param acoustic_model: Configuration of the acoustic model
    :param optimizer: Configuration for the optimizer
    :param loss: Configuration for the loss function
    :param early_stopping_patience: The number of steps with no improvement of validation loss before stopping training early
    :param batching_mode: Represents the type of batching used which changes the unit represented by the provided batch size
    :param language_oversampling_factor: Factor to use for oversampling from lower resource languages when generting batches for training
    :param seed: The random seed used for all otherwise non-deterministic operations
    :param maximum_iterations: Maximum number of epochs before the training is terminated
        If `None` is given, it acts as infinity
    :param clip_norm: Norm used for gradient clipping
    :param lr_schedule: Configuration for a learning rate schedule
    :param lr_warmup: Configuration for learning rate warmup
    :param accumulation_factor: Factor for dividing batches into mini-batches
        with gradient accumulation to allow for larger batch sizes with less memory
    :param step_size: Number of batch updates before performing validation for very large
        datasets. If `None` is given, validation is only performed once after each epoch
    :param mixed_precision: Enables efficient mixed precision training with auto scaling
    """

    batch_size: int
    projection: ProjectionConfig
    acoustic_model: AcousticModel = _nested_field(AcousticModelSchema)
    optimizer: Optimizer = _nested_field(OptimizerSchema)
    loss: CTCLossConfig = _nested_field(MainLossSchema)
    early_stopping_patience: Optional[int] = None
    batching_mode: BatchingMode = utils.schema_field(
        EnumField(BatchingMode, by_value=True), default=BatchingMode.FRAMES
    )
    language_oversampling_factor: Optional[float] = None
    seed: Optional[int] = None
    maximum_iterations: Optional[int] = None
    clip_norm: Optional[float] = None
    lr_schedule: Optional[LrSchedulerConfig] = _nested_field(LrSchedulerSchema, {"allow_none": True}, default=None)
    accumulation_factor: int = 1
    step_size: Optional[int] = None
    mixed_precision: bool = False


class FeatureType(Enum):
    """
    Type of acoustic features used by the network
    """

    MFCC = "MFCC"
    FILTERBANKS = "Filterbanks"
    RAW = "raw"


@dataclass
class Preprocessing:
    """
    Confguration for audio preprocessing

    :param window: Window for which the given `feature_type` is computed
    :param feature_type: Type of acoustic features to compute during preprocessing
    :param resample: Sampling rate to resample audio to before computing acoustic features
    :param num_filters: Number of filters for the given `feature_type`
    """

    window: Window
    feature_type: FeatureType = utils.schema_field(EnumField(FeatureType))
    resample: Optional[int] = 16_000  # 16kHz
    num_filters: int = 40


@dataclass
class DataConfig:
    """
    Configuration for the training data

    :param languages: List of languages to use for training from a multilingual corpus
    :param validation_limits: Limits the amount of utterances used as
        validation data, either a map from language codes to language specific limits,
        a single limit for every language or `None` for no limit
    :param only_primary_script: Keeps only transcriptions that consist of a
        single script that corresponds to the primary script of each language
    """

    languages: List[str]
    validation_limits: Dict[str, int] | int | None = None
    only_primary_script: bool = False


@dataclass
class ProfilingConfig:
    """
    Configuration for performance and memory profiling of training code

    :param active_steps: Number of batch update steps for which the profiler will collect samples
    :param flame_graph_path_cpu: Target path for the cpu traces
    :param flame_graph_path_gpu: Target path for the gpu traces
    :param tensorboard_dir: Path to a tensorboard log directory that profiling information will be stored in.
        When this isn't `None`, flame_graph_path_* variables will be ignored.
    :param profile_memory: Whether to profile memory consumption
    :param record_shapes: Whether to record tensors shapes during pytorch computations
    :param repeat: Number of epochs for which to repeat profiling
    """

    active_steps: int
    flame_graph_path_gpu: Optional[str] = None
    flame_graph_path_cpu: Optional[str] = None
    tensorboard_dir: Optional[str] = None
    profile_memory: bool = False
    record_shapes: bool = False
    repeat: int = 1


ConfigCls = TypeVar("ConfigCls", bound="Config")


@marshmallow_dataclass.dataclass
class Config:
    """
    Configuration for preprocessing, training and the architecture of an acoustic model

    :param nn: Configuration for the neural network architecture
    :param preprocessing: Configuration for the preprocessing of acoustic features
    :param data: Configuration for the training data
    :param profiling: Configuration for performance and memory profiling of training code
    """

    Schema: ClassVar[Type[Schema]]

    nn: Architecture
    preprocessing: Preprocessing
    data: DataConfig
    profiling: Optional[ProfilingConfig] = None

    @classmethod
    def load(cls: Type[ConfigCls], mapping: Mapping[str, Any]) -> ConfigCls:
        return cls.Schema().load(mapping)  # type: ignore

    @classmethod
    def from_toml(cls: Type[ConfigCls], path: str) -> ConfigCls:
        return cls.Schema().load(toml.load(path))  # type: ignore

    def dump(self) -> Dict[str, Any]:
        return self.Schema().dump(self)  # type: ignore

    def dumps_toml(self) -> str:
        return toml.dumps(self.dump())

    def dumps(self) -> str:
        return self.Schema().dumps(self)
