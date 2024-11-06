from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Dict, Tuple, TypeVar, Type, List, Optional
import math
import typing

import torch
from torch import LongTensor, autograd, nn
from torch.nn import functional
from torch import Tensor
from torch.nn.parameter import Parameter
from transformers import Wav2Vec2Config
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model
from transformers.models.wav2vec2.feature_extraction_wav2vec2 import Wav2Vec2FeatureExtractor

from allophant.attribute_graph import AttributeGraph, AttributeNode
from allophant.config import (
    Architecture,
    EmbeddingCompositionConfig,
    PhonemeLayerType,
    ProjectionConfig,
    ProjectionEntryConfig,
    TransformerAcousticModelConfig,
    UnfreezeScheduleConfig,
    Wav2Vec2PretrainedConfig,
)
from allophant.network import frontend
from allophant.network.frontend import Frontend, SequentialFrontend, frontend_from_config
from allophant.batching import Batch
from allophant import MAIN_LOGGER, utils
from allophant.phonetic_features import ArticulatoryAttributes, LanguageAllophoneMappings, PhoneticAttributeIndexer


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal positional embeddings from Vaswani et al. (2017). Adapted from
    the implementation by Rush (2018)

    .. references::
        Vaswani, Ashish, Noam M. Shazeer, Niki Parmar, Jakob Uszkoreit, Llion
        Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin. “Attention
        is All you Need.” Neural Information Processing Systems (2017).

        Rush, Alexander. “The Annotated Transformer.” In Proceedings of
        Workshop for NLP Open Source Software (NLP-OSS), 52–60. Melbourne,
        Australia: Association for Computational Linguistics, 2018.
        https://doi.org/10.18653/v1/W18-2509.
    """

    _LOG_10000 = math.log(10000)

    _bases: Tensor

    def __init__(self, input_size: int):
        super().__init__()
        component = torch.exp(torch.arange(0, input_size, 2, dtype=torch.float) * -(self._LOG_10000 / input_size))
        self.register_buffer("_bases", torch.stack([component] * 2, 1).view(-1), persistent=False)
        self.embedding_size = input_size

    def get_positions(self, max_positions: int) -> Tensor:
        device = self._bases.device
        position_embeddings = torch.arange(max_positions, dtype=torch.float, device=device).unsqueeze(1) * self._bases
        position_embeddings[:, 0::2] = torch.sin(position_embeddings[:, 0::2])
        position_embeddings[:, 1::2] = torch.cos(position_embeddings[:, 1::2])

        return position_embeddings.view(max_positions, -1)

    def forward(self, batch: Tensor) -> Tensor:
        return batch + self.get_positions(batch.size(0)).unsqueeze(1)


_PAD_VALUE = torch.finfo(torch.float32).min


@torch.jit.script  # pyright: ignore
def _multiply_allophone_matrix(
    phone_logits: Tensor, matrix: Tensor, mask: Tensor, mask_value: float = _PAD_VALUE
) -> Tensor:
    return (
        (phone_logits * matrix.unsqueeze(0))
        .masked_fill_(
            mask.unsqueeze(0),
            mask_value,
        )
        .max(1)
        .values
    )


class AllophoneMapping(nn.Module):
    """
    Allophone layer derived from the Allosaurus architecture (Li et al., 2020).

    .. references::
        Li, Xinjian, Siddharth Dalmia, Juncheng Li, Matthew Lee, Patrick
        Littell, Jiali Yao, Antonios Anastasopoulos, et al. “Universal Phone
        Recognition with a Multilingual Allophone System.” In ICASSP 2020 -
        2020 IEEE International Conference on Acoustics, Speech and Signal
        Processing (ICASSP), 8249–53. Barcelona, Spain: IEEE, 2020.
        https://doi.org/10.1109/ICASSP40776.2020.9054362.
    """

    _allophone_mask: Tensor

    def __init__(
        self,
        shared_phone_count: int,
        phoneme_count: int,
        blank_offset: int,
        language_allophones: LanguageAllophoneMappings,
    ) -> None:
        super().__init__()
        allophones = language_allophones.allophones
        languages = language_allophones.languages
        num_languages = len(languages)
        # Maps language codes to dense indices in the matrix
        self._index_map = {}

        allophone_matrix = torch.zeros(num_languages, shared_phone_count, phoneme_count)
        for dense_index, (language_index, allophone_mapping) in enumerate(allophones.items()):
            language_allophone_matrix = allophone_matrix[dense_index]
            # Constructs identity mappings for BLANK in the diagonal
            language_allophone_matrix[range(blank_offset), range(blank_offset)] = 1

            self._index_map[languages[language_index]] = dense_index
            for phoneme, allophones in allophone_mapping.items():
                # Add allophone mappings with the blank offset
                language_allophone_matrix[torch.tensor(allophones) + blank_offset, phoneme + blank_offset] = 1

        # Shared allophone_matrix
        self._allophone_matrices = Parameter(allophone_matrix)

        # Initialization for the l2 penalty according to "Universal Phone Recognition With a Multilingual Allophone Systems" by Li et al.
        self.register_buffer("_initialization", allophone_matrix.clone(), persistent=False)
        # Inverse mask for language specific phoneme inventories
        self.register_buffer("_allophone_mask", ~allophone_matrix.bool(), persistent=False)

    @property
    def index_map(self) -> Dict[str, int]:
        return self._index_map

    def map_allophones(self, phone_logits: Tensor, language_ids: Tensor) -> Tensor:
        # Batch size x Shared Phoneme Inventory Size
        batch_matrices = torch.empty(
            *phone_logits.shape[:2], self._allophone_matrices.shape[2], device=phone_logits.device
        )
        for index, language_id in enumerate(map(int, language_ids)):
            logits = phone_logits[:, index].unsqueeze(-1)
            # Max pooling after multiplying with the allophone matrix
            # Replace masked positions with negative infinity
            # since this results in zero probabilities after softmax for phone and
            # phoneme combinations that don't occur in the allophone mappings
            batch_matrices[:, index] = _multiply_allophone_matrix(
                logits,
                self._allophone_matrices[language_id],
                self._allophone_mask[language_id],
            )

        return batch_matrices

    def forward(self, phone_logits: Tensor, language_ids: Tensor, predict: bool = False) -> Dict[str, Tensor]:
        # Note that while predictions on the training corpus could still be valid with the allophone layer enabled,
        # language IDs are different for other corpora and therefore not supported
        if predict:
            # Also assigns phone logits to the phoneme layer in case another layer uses phonemes as a dependency
            return {ProjectionEntryConfig.PHONE: phone_logits, ProjectionEntryConfig.PHONEME_LAYER: phone_logits}
        return {ProjectionEntryConfig.PHONEME_LAYER: self.map_allophones(phone_logits, language_ids)}

    def l2_penalty(self) -> Tensor:
        """
        Computes the l2 penalty for the allophone layer to be added to the loss

        :return: The l2 penalty for the allophone layer given the current
            weights or 0 if no allophone layer is present in the architecture
        """
        # Calculates the matrix L2 (Frobenius) norm for each allophone matrix and then sums the norms over languages
        return torch.norm_except_dim(self._allophone_matrices - self._initialization, dim=0).sum()


class EmbeddingCompositionLayer(nn.Module):
    """
    Embedding composition layer derived from the compositional phone embedding
    layer by Li et al, (2021)

    .. references::
        Li, Xinjian, Juncheng Li, Florian Metze and Alan W. Black.
        “Hierarchical Phone Recognition with Compositional Phonetics.”
        Interspeech (2021).
    """

    def __init__(self, embedding_size: int, attribute_indexer: ArticulatoryAttributes) -> None:
        super().__init__()

        dense_feature_table = attribute_indexer.dense_feature_table.long()
        # Add Single blank embedding
        num_categories = torch.cat((LongTensor([0]), dense_feature_table.max(0).values)) + 1
        unused_categories = torch.cat(
            (torch.tensor([False]), torch.cat([row.bincount() for row in dense_feature_table.T]) == 0)
        )

        unused_count = unused_categories.sum().item()
        if unused_count:
            MAIN_LOGGER.info(f"{unused_count} unused feature embeddings")

        # Offsets and one additional entry for the special blank feature
        category_offsets = num_categories.cumsum(0)[:-1].unsqueeze(0)
        dense_feature_table += category_offsets
        self._attribute_embeddings = nn.EmbeddingBag(int(num_categories.sum()), embedding_size, mode="sum")

        # Set unused attribute embedding weights to 0
        with torch.no_grad():
            self._attribute_embeddings.weight[unused_categories] = 0

        self.register_buffer("_dense_feature_table", dense_feature_table, persistent=False)
        self.register_buffer("_category_offsets", category_offsets, persistent=False)
        # Scale factor for the dot product with feature embeddings as in Li et al. (2021)
        self.register_buffer("_scale_factor", torch.tensor(math.sqrt(embedding_size)), persistent=False)

    def forward(self, inputs: Tensor, target_feature_indices: Tensor | None = None) -> Tensor:
        if target_feature_indices is None:
            target_feature_indices = typing.cast(Tensor, self._dense_feature_table)
        else:
            target_feature_indices = target_feature_indices + self._category_offsets

        composed_embeddings = torch.cat(
            (
                # Blank embedding (Using the index batch sequence equivalent to [[0]])
                self._attribute_embeddings(torch.zeros(1, 1, dtype=target_feature_indices.dtype, device=inputs.device)),
                # Phonemes
                self._attribute_embeddings(target_feature_indices),
            )
        ).T

        return (inputs @ composed_embeddings) / self._scale_factor


class ProjectingMultiheadAttention(nn.Module):
    def __init__(
        self,
        input_dimensions: int,
        hidden_dimensions: int,
        num_heads: int,
        add_positional_embeddings: bool = False,
        dropout_rate: float = 0,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dimensions, hidden_dimensions)
        self.positional_embeddings = (
            SinusoidalPositionEmbeddings(hidden_dimensions) if add_positional_embeddings else None
        )
        self.layer_norm = nn.LayerNorm(hidden_dimensions)
        self.attention = nn.MultiheadAttention(hidden_dimensions, num_heads)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs: Tensor, lengths: Tensor) -> Tensor:
        projection_output = self.layer_norm(self.input_projection(inputs))
        if self.positional_embeddings is not None:
            projection_output = self.positional_embeddings(projection_output)

        return self.dropout(
            self.attention(
                projection_output,
                projection_output,
                projection_output,
                utils.mask_sequence(lengths, inverse=True),
                need_weights=False,
            )[0]
        )


class HierarchicalClassifier(nn.Module):
    def __init__(
        self,
        time_distributed_layer: ProjectingMultiheadAttention | nn.Linear,
        composition_layer: EmbeddingCompositionLayer | None = None,
        allophone_layer: AllophoneMapping | None = None,
    ) -> None:
        super().__init__()
        self._lengths_required = not isinstance(time_distributed_layer, nn.Linear)
        self._time_distributed_layer = time_distributed_layer
        self._composition_layer = composition_layer
        self._allophone_layer = allophone_layer

    def forward(
        self,
        inputs: Tensor,
        lengths: Tensor,
        language_ids: Tensor,
        target_feature_indices: Tensor | None = None,
        predict: bool = False,
    ) -> Tensor:
        if self._lengths_required:
            inputs = self._time_distributed_layer(
                inputs,
                lengths,
            )
        else:
            inputs = self._time_distributed_layer(inputs)

        if self._composition_layer is not None:
            inputs = self._composition_layer(inputs, target_feature_indices)

        if self._allophone_layer is None:
            return inputs

        return self._allophone_layer(inputs, language_ids, predict)


def _process_classifier_dependencies(
    attribute_graph: AttributeGraph,
    node: AttributeNode,
    output_features: int,
    blank_offset: int,
    dependency_blanks: bool = True,
) -> Tuple[int, List[AttributeNode]]:
    layer_input_neurons = 0
    dependencies = []
    for target in node.dependencies:
        attribute_node = attribute_graph.get(target)
        if attribute_node is None:
            # Output nodes aren't in the graph and are turned into a node for convenience
            attribute_node = AttributeNode(target, output_features)
        elif dependency_blanks:
            # Add the blank offset only to phonetic attribute nodes if necessary
            attribute_node = attribute_node.with_offset(blank_offset)

        layer_input_neurons += attribute_node.size
        dependencies.append(attribute_node)

    return layer_input_neurons, dependencies


class HierarchicalProjection(nn.Module):
    _OUTPUT_PATTERN = ProjectionEntryConfig.OUTPUT_PATTERN

    def __init__(
        self,
        output_features: int,
        attribute_graph: AttributeGraph,
        blank_offset: int,
        dependency_blanks: bool = True,
        language_allophones: LanguageAllophoneMappings | None = None,
        attribute_indexer: PhoneticAttributeIndexer | None = None,
        acoustic_model_dropout_rate: float = 0,
        embedding_composition_config: EmbeddingCompositionConfig | None = None,
    ):
        super().__init__()
        self._acoustic_model_dropout = (
            nn.Dropout(acoustic_model_dropout_rate) if acoustic_model_dropout_rate > 0 else None
        )
        self._uses_allophone_mapping = False

        dependency_names = set(attribute_graph.names())
        if len(dependency_names) < len(attribute_graph):
            raise ValueError("Dependencies contain duplicate keys")
        if any(self._OUTPUT_PATTERN.match(name) for name in dependency_names):
            raise ValueError(f"{ProjectionEntryConfig.OUTPUT_DEPENDENCY!r} is a reserved keyword")

        self._blank_offset = blank_offset
        self._dependency_blanks = dependency_blanks

        self._layers = nn.ModuleDict()
        self._ordered_nodes = []

        required_output_layers = set()

        # Generate layers by iterating over the attribute graph in reverse topological order
        for node in attribute_graph.sort():
            # Retrieve dependency nodes
            layer_input_neurons, dependencies = _process_classifier_dependencies(
                attribute_graph,
                node,
                output_features,
                blank_offset,
                dependency_blanks,
            )

            if not dependencies:
                raise ValueError("Each class projection requires a dependency")

            # Keep track of topologically ordered graph nodes with their dependencies for the forward pass
            self._ordered_nodes.append((node.name, dependencies))
            # Keep track of required acoustic model output layers
            required_output_layers.update(
                dependency.name for dependency in dependencies if self._OUTPUT_PATTERN.match(dependency.name)
            )

            # Use an allophone layer
            is_phoneme_layer = node.name == ProjectionEntryConfig.PHONEME_LAYER
            if language_allophones is not None and is_phoneme_layer:
                self._uses_allophone_mapping = True
                # Use the size of the phone inventory as an input including CTC
                # offset as passthrough to allow for decoding even though shared phones aren't explicitly supervised
                output_size = len(language_allophones.shared_phones) + blank_offset
            else:
                # Get the output size plus space for CTC blanks
                output_size = node.size + blank_offset

            if is_phoneme_layer and embedding_composition_config is not None:
                # Project to the embedding size when using embedding composition
                projection_output_size = embedding_composition_config.embedding_size
            else:
                # Project to the number of classes otherwise
                projection_output_size = output_size

            if node.time_layer_config is not None:
                time_distributed_layer = ProjectingMultiheadAttention(
                    layer_input_neurons,
                    projection_output_size,
                    node.time_layer_config.num_heads,
                    node.time_layer_config.positional_embeddings,
                    acoustic_model_dropout_rate,
                )
            else:
                time_distributed_layer = nn.Linear(layer_input_neurons, projection_output_size)

            if is_phoneme_layer and embedding_composition_config is not None:
                if attribute_indexer is None:
                    raise ValueError(
                        "Model configuration using attribute embedding composition"
                        " requires an attribute indexer but got `None`"
                    )

                if not self._uses_allophone_mapping:
                    # Use the phoneme subset from the attribute indexer with all features for constructing the embeddings
                    training_attributes = attribute_indexer.full_attributes.subset(
                        attribute_indexer.phonemes.tolist(),
                        attribute_indexer.composition_features.copy(),
                    )
                else:
                    if attribute_indexer.allophone_data is None:
                        raise ValueError(
                            "Model configuration using attribute embedding composition and an allophone layer"
                            " requires allophone data in the attribute indexer with but got `None`"
                        )

                    training_attributes = attribute_indexer.allophone_data.shared_phone_indexer

                if output_size != len(training_attributes) + 1:
                    raise ValueError(
                        f"Length of attributes with blanks ({len(training_attributes) + 1}) need to match"
                        f" the number of phones in the allophone mapping ({output_size})"
                    )

                embedding_size = embedding_composition_config.embedding_size
                composition_layer = EmbeddingCompositionLayer(embedding_size, training_attributes)
            else:
                composition_layer = None

            if is_phoneme_layer and self._uses_allophone_mapping:
                allophone_layer = AllophoneMapping(
                    output_size,
                    # The node contains the number of language specific phonemes
                    node.size + blank_offset,
                    blank_offset,
                    typing.cast(LanguageAllophoneMappings, language_allophones),
                )
            else:
                allophone_layer = None

            self._layers[node.name] = HierarchicalClassifier(time_distributed_layer, composition_layer, allophone_layer)

        if not required_output_layers:
            raise ValueError(
                "At least one of the input layers requires {ProjectionEntryConfig.OUTPUT_DEPENDENCY!r} as a dependency"
            )

        # Sorted for consistency
        self._output_dependencies = sorted(required_output_layers)

    def forward(
        self,
        inputs: List[Tensor],
        input_lengths: Tensor,
        language_ids: Tensor,
        target_feature_indices: Tensor | None = None,
        predict: bool = False,
    ) -> Dict[str, Tensor]:
        outputs = {
            f"{ProjectionEntryConfig.OUTPUT_DEPENDENCY}_{index}": input_layer
            for index, input_layer in enumerate(inputs)
        }
        outputs[ProjectionEntryConfig.OUTPUT_DEPENDENCY] = inputs[-1]

        # Apply dropout to the acoustic model outputs if necessary
        if self._acoustic_model_dropout is not None:
            for dependency in self._output_dependencies:
                outputs[dependency] = self._acoustic_model_dropout(outputs[dependency])

        projection_outputs = {}
        # Iterate over ordered nodes and layer - note that ModuleDict maintains insertion order
        for (name, dependencies), layer in zip(self._ordered_nodes, self._layers.values()):
            # Fast path for a single output dependency to avoid concatenation
            if len(dependencies) == 1 and self._OUTPUT_PATTERN.match(dependencies[0].name):
                dependency_outputs = outputs[dependencies[0].name]
            else:
                dependency_outputs = torch.cat(
                    [
                        (
                            torch.softmax(
                                (
                                    outputs[dependency.name]
                                    if self._dependency_blanks
                                    else outputs[dependency.name][..., self._blank_offset :]
                                ),
                                -1,
                            )
                            if not self._OUTPUT_PATTERN.match(dependency.name)
                            else outputs[dependency.name]
                        )
                        for dependency in dependencies
                    ],
                    -1,
                )

            output = layer(dependency_outputs, input_lengths, language_ids, target_feature_indices, predict)
            if isinstance(output, dict):
                projection_outputs.update(output)
                outputs.update(output)
            else:
                projection_outputs[name] = output
                outputs[name] = output

        return projection_outputs

    def l2_penalty(self) -> Tensor | None:
        """
        Computes the l2 penalty for the allophone layer to be added to the loss

        :return: The l2 penalty for the allophone layer given the current
            weights or 0 if no allophone layer is present in the architecture
        """
        try:
            allophone_mapping = self._layers[ProjectionEntryConfig.PHONEME_LAYER]
            if isinstance(allophone_mapping, AllophoneMapping):
                return allophone_mapping.l2_penalty()
        finally:
            # No penalty if there's no allophone layer in the architecture
            return None

    def map_allophones(self, phone_logits: Tensor, language_ids: Tensor) -> Tensor:
        allophone_mapping = self._layers[ProjectionEntryConfig.PHONEME_LAYER]
        if isinstance(allophone_mapping, AllophoneMapping):
            return allophone_mapping.map_allophones(phone_logits, language_ids)

        raise ValueError("Can't map phones to allophones with a model without an allophone layer")

    @property
    def classifier_layers(self) -> nn.ModuleDict:
        return self._layers


# NOTE: Taken from https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerEncoderLayer
def _get_activation_fn(activation):
    if activation == "relu":
        return functional.relu
    elif activation == "gelu":
        return functional.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


# NOTE: Modified from https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerEncoderLayer
class PreLMTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        elementwise_affine: bool = False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=elementwise_affine)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=elementwise_affine)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = functional.relu
        super().__setstate__(state)

    def forward(
        self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None
    ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src = self.norm1(src)
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(self.norm2(src)))))
        src = src + self.dropout2(src2)
        return src


class TransformerEncoderIntermediate(nn.TransformerEncoder):
    def __self__(self, encoder_layer: nn.TransformerEncoderLayer, num_layers: int):
        super().__init__(encoder_layer, num_layers)
        super().forward

    def forward(
        self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None
    ) -> List[Tensor]:
        output = src
        outputs = []

        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            outputs.append(output)

        if self.norm is not None:
            output = self.norm(output)
            outputs[-1] = output

        return outputs


class AcousticModel(nn.Module, metaclass=ABCMeta):
    _feature_size: int
    _output_size: int
    _upscale_factor: float
    _d_model: int

    @property
    def feature_size(self) -> int:
        return self._feature_size

    @property
    def output_size(self) -> int:
        return self._output_size

    @property
    def d_model(self) -> int:
        return self._d_model

    @property
    def upscale_factor(self) -> float:
        return self._upscale_factor

    @abstractmethod
    def downsampled_lengths(self, lengths: Tensor) -> Tensor: ...

    @abstractmethod
    def forward(self, batch: Batch, _predict: bool = False) -> Tuple[List[Tensor], Tensor]: ...


class TransformerAcousticModel(AcousticModel):
    def __init__(
        self,
        frontend: Frontend,
        transformer: TransformerEncoderIntermediate,
        sequential_frontend: Optional[SequentialFrontend] = None,
        input_dropout_rate: float = 0,
        use_positional_embeddings: bool = True,
        elementwise_affine: bool = False,
    ) -> None:
        super().__init__()
        self._input_dropout = nn.Dropout(input_dropout_rate)
        self._frontend = frontend
        self._transformer = transformer
        self._feature_size = frontend.output_dimensions
        self._final_layer_norm = nn.LayerNorm(
            self._feature_size if sequential_frontend is None else sequential_frontend.output_dimensions,
            elementwise_affine=elementwise_affine,
        )
        self._positional_embeddings = (
            SinusoidalPositionEmbeddings(
                self._feature_size if sequential_frontend is None else sequential_frontend.output_dimensions
            )
            if use_positional_embeddings
            else None
        )
        self._sequential_frontend = sequential_frontend
        self._upscale_factor = 1 if sequential_frontend is None else sequential_frontend.upscale_factor
        self._d_model = typing.cast(nn.Linear, transformer.layers[0].linear1).in_features
        self._output_size = typing.cast(nn.Linear, transformer.layers[-1].linear2).out_features

    @property
    def feature_size(self) -> int:
        return self._feature_size

    @property
    def upscale_factor(self) -> float:
        return self._upscale_factor

    def forward(self, batch: Batch, _predict: bool = False) -> Tuple[List[Tensor], Tensor]:
        frontend_output = self._frontend(batch)
        # Permutes from (N, F, L) to (L, N, F) for transformer (L = sequence, N = batch, F = features)
        frontend_features = frontend_output.audio_features.permute(2, 0, 1)
        model_inputs = self._input_dropout(frontend_features)

        if self._sequential_frontend is not None:
            frontend_output = self._sequential_frontend(
                Batch(model_inputs.permute(1, 2, 0), frontend_output.lengths, frontend_output.language_ids)
            )
            model_inputs = frontend_output.audio_features.permute(2, 0, 1)

        transformer_output = self._transformer(
            self._positional_embeddings(model_inputs) if self._positional_embeddings is not None else model_inputs,
            # Inverse mask since TransformerEncoderLayer expects `True` in padding locations
            src_key_padding_mask=utils.mask_sequence(frontend_output.lengths, inverse=True),
        )

        return [self._final_layer_norm(output) for output in transformer_output], frontend_output.lengths

    def downsampled_lengths(self, lengths: Tensor) -> Tensor:
        lengths = self._frontend.lengths(lengths)
        if self._sequential_frontend is None:
            return lengths
        return self._sequential_frontend.downsampled_lengths(lengths)

    @classmethod
    def from_config(cls, layer_config: TransformerAcousticModelConfig, feature_size: int):
        transformer_config = layer_config.transformer
        frontend = frontend_from_config(layer_config.frontend, feature_size, layer_config.elementwise_affine)
        previous_output_size = frontend.output_dimensions
        sequential_frontend_config = layer_config.sequential_frontend
        if sequential_frontend_config is not None:
            sequential_frontend = SequentialFrontend.from_config(sequential_frontend_config, previous_output_size)
            previous_output_size = sequential_frontend.output_dimensions
        else:
            sequential_frontend = None

        return cls(
            frontend,
            TransformerEncoderIntermediate(
                PreLMTransformerEncoderLayer(
                    previous_output_size,
                    transformer_config.heads,
                    transformer_config.feedforward_neurons,
                    transformer_config.dropout_rate,
                    transformer_config.activation,
                    layer_config.elementwise_affine,
                ),
                transformer_config.num_layers,
            ),
            sequential_frontend,
            transformer_config.dropout_rate,
            transformer_config.positional_embeddings,
            layer_config.elementwise_affine,
        )


@torch.jit.script
def zero_mean_unit_var_norm(features: Tensor, lengths: Tensor, mask: Tensor) -> Tensor:
    means = (features.sum(1) / lengths).unsqueeze(1)
    deviations = (features - means) * mask
    variances = (deviations**2).sum(1) / lengths
    return ((features - means) / (variances.unsqueeze(1) + 1e-7).sqrt()) * mask


def _freeze_module(model: nn.Module, trainable: bool = False) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = trainable


class Wav2Vec2AcousticModel(AcousticModel):
    def __init__(
        self,
        model_id: str,
        sampling_rate: int = 16_000,
        freeze_feature_encoder: bool = True,
        freeze_feature_projection: bool = False,
        freeze_encoder: bool = False,
        load_pretrained_weights: bool = True,
        maximum_encoder_layers: int | None = None,
    ) -> None:
        super().__init__()
        self._feature_options, _ = Wav2Vec2FeatureExtractor.get_feature_extractor_dict(model_id)
        expected_sampling_rate = self._feature_options["sampling_rate"]
        if sampling_rate != expected_sampling_rate:
            raise ValueError(
                "Audio resampling config and the sampling rate required by Wav2Vec2 do not match. "
                f"Expected {expected_sampling_rate}kHz, got {sampling_rate}kHz"
            )

        if load_pretrained_weights:
            self._model = typing.cast(Wav2Vec2Model, Wav2Vec2Model.from_pretrained(model_id))
        else:
            self._model = Wav2Vec2Model(typing.cast(Wav2Vec2Config, Wav2Vec2Config.from_pretrained(model_id)))

        # Slice encoder layers if requested to save memory and compute
        if maximum_encoder_layers is not None:
            self._model.encoder._layers = self._model.encoder.layers[:maximum_encoder_layers]

        self._model.train(self.training)
        # Freeze the feature encoder as in https://arxiv.org/abs/2109.11680
        if freeze_feature_encoder:
            self._model.freeze_feature_encoder()
        # Freeze feature projection and encoder manually since there is no API method for it
        if freeze_feature_projection:
            _freeze_module(self._model.feature_projection)
        if freeze_encoder:
            _freeze_module(self._model.encoder)

        self._use_attention_mask = self._feature_options["return_attention_mask"]
        self._normalize = self._feature_options["do_normalize"]
        self._feature_size = self._feature_options["feature_size"]
        self._upscale_factor = 1
        config = self._model.config
        self._d_model = config.hidden_size
        # Falls back to d_model if no adapter with a different hidden size is used
        self._output_size = config.output_hidden_size or self._d_model
        self._sampling_rate = sampling_rate
        self._length_functions = [
            frontend.conv_length(kernel_size, stride, use_padding=False)
            for kernel_size, stride in zip(config.conv_kernel, config.conv_stride)
        ]

    @property
    def model(self) -> Wav2Vec2Model:
        return self._model

    def downsampled_lengths(self, lengths: Tensor) -> Tensor:
        for convolution_function in self._length_functions:
            lengths = convolution_function(lengths)
        return lengths

    def forward(self, batch: Batch, _predict: bool = False) -> Tuple[List[Tensor], Tensor]:
        sequence_mask = utils.mask_sequence(batch.lengths)
        output_hidden_states = self._model(
            (
                zero_mean_unit_var_norm(batch.audio_features, batch.lengths, sequence_mask)
                if self._normalize
                else batch.audio_features
            ),
            sequence_mask.long() if self._use_attention_mask else None,
            output_hidden_states=True,
        ).hidden_states

        # Transpose all hidden states to time dimension first for CTC
        return (
            [hidden_state.transpose(0, 1) for hidden_state in output_hidden_states],
            self.downsampled_lengths(batch.lengths),
        )


@dataclass
class UnfreezeSchedule:
    def __init__(
        self,
        feature_extractor: int | None = None,
        feature_projection: int | None = None,
        encoder_steps_remaining: int | None = None,
    ):
        self._steps = 0
        self._steps_remaining = [
            feature_extractor,
            feature_projection,
            encoder_steps_remaining,
        ]

    def step(self, acoustic_model: AcousticModel):
        if not isinstance(acoustic_model, Wav2Vec2AcousticModel):
            raise ValueError(
                f"Found an unsupported acoustic module type while updating an unfreeze schedule: {type(acoustic_model)}"
            )

        for index, layer in enumerate(
            (
                acoustic_model._model.feature_extractor,
                acoustic_model._model.feature_projection,
                acoustic_model._model.encoder,
            )
        ):
            # Skip layers that are already unfrozen or aren't scheduled to be
            steps = self._steps_remaining[index]
            if steps is None:
                continue

            # Take a step
            steps -= 1
            # Disable schedule and unfreeze the layer if all remaining steps are complete
            if steps <= 0:
                steps = None
                _freeze_module(layer, trainable=True)

            # Update remaining steps
            self._steps_remaining[index] = steps

    @classmethod
    def from_config(cls, config: UnfreezeScheduleConfig):
        return UnfreezeSchedule(
            config.feature_encoder_steps,
            config.feature_projection_steps,
            config.encoder_steps,
        )


@dataclass
class Predictions:
    """
    Contains predictions of a list of logits for each task and the number of output frames
    for each input utterance in a batch

    :param outputs: A dictionary of named output logit or probabilitiy tensor batches for each task
    :param lengths: A batch of the number of output frames
    """

    outputs: Dict[str, Tensor]
    lengths: Tensor

    def __len__(self) -> int:
        # Number of elements per batch
        return len(self.lengths)

    def task_count(self) -> int:
        return len(self.outputs)


AllophantCls = TypeVar("AllophantCls", bound="Allophant")


def _highest_specific_output_layer(graph: AttributeGraph) -> int | None:
    output_layer_indices = []
    for node in graph:
        for dependency in node.dependencies:
            output_match = ProjectionEntryConfig.OUTPUT_PATTERN.match(dependency)
            if output_match is not None and (layer_index := output_match.group(1)) is not None:
                output_layer_indices.append(int(layer_index))

    # Get exclusive index of the highest specific output layer or None, if only the final output is included
    return max(output_layer_indices) + 1 if output_layer_indices else None


class Allophant(nn.Module):
    def __init__(
        self,
        acoustic_model: AcousticModel,
        attribute_graph: AttributeGraph,
        blank_offset: int,
        projection_config: ProjectionConfig,
        attribute_indexer: PhoneticAttributeIndexer | None = None,
    ):
        super().__init__()
        self._acoustic_model = acoustic_model
        if attribute_indexer is not None and projection_config.phoneme_layer != PhonemeLayerType.SHARED:
            language_allophones = attribute_indexer.language_allophones
        else:
            language_allophones = None

        self._projection = HierarchicalProjection(
            acoustic_model.output_size,
            attribute_graph,
            blank_offset,
            projection_config.dependency_blanks,
            language_allophones,
            attribute_indexer,
            projection_config.acoustic_model_dropout,
            projection_config.embedding_composition,
        )
        self._classes = list(attribute_graph.names())

    @property
    def acoustic_model(self) -> AcousticModel:
        return self._acoustic_model

    @property
    def d_model(self) -> int:
        return self._acoustic_model.d_model

    @property
    def feature_size(self) -> int:
        return self._acoustic_model.feature_size

    @property
    def classes(self) -> List[str]:
        return self._classes

    @classmethod
    def from_config(
        cls: Type[AllophantCls],
        architecture: Architecture,
        feature_size: int,
        sample_rate: int,
        attribute_graph: AttributeGraph,
        attribute_indexer: PhoneticAttributeIndexer | None = None,
        load_pretrained_weights: bool = True,
    ) -> AllophantCls:
        layer_config = architecture.acoustic_model
        match layer_config:
            case TransformerAcousticModelConfig():
                acoustic_model = TransformerAcousticModel.from_config(layer_config, feature_size)
            case Wav2Vec2PretrainedConfig(model_id, freeze_feature_encoder, freeze_feature_projection, freeze_encoder):
                acoustic_model = Wav2Vec2AcousticModel(
                    model_id,
                    sample_rate,
                    freeze_feature_encoder,
                    freeze_feature_projection,
                    freeze_encoder,
                    load_pretrained_weights,
                    # Limit the pre-trained wav2vec2 model to at most the
                    # highest specific output layer required by the configuration for efficiency
                    _highest_specific_output_layer(attribute_graph),
                )
            case Wav2Vec2Model():
                raise NotImplementedError("Training Wav2Vec2 from scratch is not yet implemented")
            case model:
                raise ValueError(f"Unsupported model type: {type(model)}")

        return cls(
            acoustic_model,
            attribute_graph,
            architecture.loss.BLANK_OFFSET,
            architecture.projection,
            attribute_indexer,
        )

    def forward(self, batch: Batch, target_feature_indices: Tensor | None = None, predict: bool = False) -> Predictions:
        outputs, lengths = self._acoustic_model(batch, predict)
        return Predictions(
            self._projection(
                outputs,
                lengths,
                batch.language_ids,
                target_feature_indices,
                predict,
            ),
            lengths,
        )

    def map_allophones(self, phone_logits: Tensor, language_ids: Tensor) -> Tensor:
        return self._projection.map_allophones(phone_logits, language_ids)

    @property
    def upscale_factor(self) -> float:
        return self._acoustic_model.upscale_factor

    @property
    def projection(self) -> HierarchicalProjection:
        return self._projection

    def log_probabilities(self, outputs: Tensor) -> Tensor:
        return functional.log_softmax(outputs, -1)

    def downsampled_lengths(self, lengths: Tensor) -> Tensor:
        return self._acoustic_model.downsampled_lengths(lengths)

    def l2_penalty(self) -> Tensor | None:
        """
        Computes the l2 penalty for the allophone layer to be added to the loss

        :return: The l2 penalty for the allophone layer given the current
            weights or `None` if no allophone layer is present in the architecture
        """
        return self._projection.l2_penalty()
