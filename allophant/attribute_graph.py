from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type
import typing

from marshmallow import Schema
from marshmallow.fields import Field
import marshmallow_dataclass

from allophant.config import MultiheadAttentionConfig, ProjectionEntryConfig


class DependencyCycleError(Exception):
    """Raised when a dependency cycle is detected"""


@dataclass
class AttributeNode:
    """
    Contains the name, number of variants and dependencies of a phonetic attribute node

    :param name: Name of the attribute
    :param size: Number of variants of the attribute
    :param dependencies: Attribute names of each dependency
    """

    name: str
    size: int
    time_layer_config: Optional[MultiheadAttentionConfig] = None
    dependencies: List[str] = field(default_factory=list)

    def with_offset(self, offset: int = 1):
        """
        Creates a new phonetic attribute node with an offset added to the size
        but the same name and dependencies

        :param offset: An offset that is added to the node size

        :return: A new phonetic attribute node derived from the current node with `offset` added to its size
        """
        return AttributeNode(self.name, self.size + offset, self.time_layer_config, self.dependencies)


class AttributeGraph:
    """
    A graph of phonetic attributes which also stores the number of variants of
    each attribute and provides methods for topological sorting
    """

    _nodes: List[AttributeNode]
    _node_indices: Dict[str, int]
    _edges: List[List[int]]

    def __init__(self, nodes: Iterable[AttributeNode]) -> None:
        """
        Builds an attribute graph from a sequence of attributes with their number of variants and dependencies

        :param nodes: An iterable of attribute, size and dependencies tuples
        """
        self._nodes = []
        self._node_indices = {}

        for index, node in enumerate(nodes):
            self._nodes.append(node)
            self._node_indices[node.name] = index

        self._edges = [
            [
                self._node_indices[dependency]
                for dependency in node.dependencies
                if not ProjectionEntryConfig.OUTPUT_PATTERN.match(dependency)
            ]
            for node in self._nodes
        ]

    def sizes(self) -> Iterator[int]:
        """
        Generates the number of variants for each node in the graph

        :return: A generator over the number of variants for each node
        """
        return (node.size for node in self._nodes)

    def names(self) -> Iterator[str]:
        """
        Generates the phonetic attribute names for each node in the graph

        :return: A generator over the phonetic attribute names for each node
        """
        return (node.name for node in self._nodes)

    @property
    def nodes(self) -> List[AttributeNode]:
        return self._nodes

    def get(self, node: str | int) -> AttributeNode | None:
        """
        Retrieves a phonetic attribute node by either its name or its index in the graph

        :param node: The name or index of a phonetic attribute node

        :return: The attribute node with name or index `node` or `None` if a
            name was given but not found in the graph
        """
        if isinstance(node, str):
            node_index = self._node_indices.get(node)
            if node_index is None:
                return None
            node = node_index

        return self._nodes[node]

    def __getitem__(self, node: str | int) -> AttributeNode:
        if isinstance(node, str):
            node = self._node_indices[node]
        return self._nodes[node]

    def __contains__(self, node_name: str) -> bool:
        return node_name in self._node_indices

    def __len__(self) -> int:
        return len(self._nodes)

    def strongly_connected_components(self) -> Iterator[List[AttributeNode]]:
        """
        Yields the strongly connected components (SCC) of the graph in reverse topological order using Tarjan's SCC algorithm.
        Components with more than one element indicate a cycle and contain every node that participates in it.

        :return: A generator over all strongly connected components of the graph, represented by lists of :py:class:`AttributeNode` instances
        """
        # Immediately return on empty node
        if not self._nodes:
            return

        stack: List[int] = []
        call_stack: List[Tuple[int, int, int]] = [(0, 0, 0)]
        visited_count = 0
        low_map = {}
        remaining_nodes = iter(range(1, len(self)))

        while True:
            if call_stack:
                node, index, lowlink = call_stack.pop()
            else:
                # If the call stack is empty start the next traversal in case
                # the graph is disconnected
                try:
                    node = next(remaining_nodes)
                except StopIteration:
                    # End traversal if there are no remaining nodes
                    return
                index = 0
                lowlink = visited_count

            if index == 0:
                # Handle the first edge of a node if it wasn't already visited
                if node in low_map:
                    continue
                low_map[node] = lowlink
                visited_count += 1
                stack.append(node)
            else:
                low_map[node] = min(low_map[node], low_map[self._edges[node][index - 1]])

            if index < len(self._edges[node]):
                # If there are more outgoing edges from the current node, first
                # visit the next edge and then the target of the current edge
                call_stack.extend(
                    (
                        (node, index + 1, lowlink),
                        (self._edges[node][index], 0, visited_count),
                    )
                )
                continue

            # If a component is connected, reconstruct and return the current
            # strongly connected component by backtracking
            if lowlink == low_map[node]:
                components = []
                while stack:
                    stack_node = stack.pop()
                    components.append(stack_node)
                    low_map[stack_node] = len(self)
                    if stack_node == node:
                        break

                # If this list has a length of one, the containing node doesn't participate in a cycle
                yield [self._nodes[index] for index in components]

    def sort(self) -> Iterator[AttributeNode]:
        """
        Sorts the graph in reverse topological order and raises a :py:exc:`DependencyCycleError` if dependency cycles are detected

        :return: A generator over :py:class:`AttributeNode` instance in reverse topological order according to their dependencies
        """
        for component in self.strongly_connected_components():
            if len(component) > 1:
                raise DependencyCycleError(f"Dependency cycle detected: {' -> '.join(node.name for node in component)}")
            yield component[0]


@marshmallow_dataclass.dataclass
class _AttributeGraphSchema:
    """
    Dataclass schema for :py:class:`AttributeGraph` serialization

    :param nodes: The phonetic attribute nodes of the graph
    :param node_indices: A mapping of node names to node indices
    :param edges: A mapping of node indices to the indices of their dependents
    """

    Schema: ClassVar[Type[Schema]]

    nodes: List[AttributeNode]
    node_indices: Dict[str, int]
    edges: List[List[int]]


class AttributeGraphField(Field):
    """
    :py:mod:`marshmallow` field for serializing and deserializing an :py:class:`AttributeGraph`
    """

    _Schema = _AttributeGraphSchema.Schema()

    def _serialize(self, value: AttributeGraph, _attr: str, _obj: Any, **kwargs) -> Dict[str, Any]:
        return self._Schema.dump(
            _AttributeGraphSchema(  # type: ignore
                value._nodes,
                value._node_indices,
                value._edges,
            )
        )

    def _deserialize(self, value: Any, _attr: str | None, _data: Mapping[str, Any] | None, **kwargs) -> AttributeGraph:
        fields = typing.cast(_AttributeGraphSchema, self._Schema.load(value))
        # Manually initializes the graph from the deserialized fields
        graph = AttributeGraph.__new__(AttributeGraph)
        graph._nodes = fields.nodes
        graph._node_indices = fields.node_indices
        graph._edges = fields.edges
        return graph
