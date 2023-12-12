"""
Base classes for graph nodes.
"""

import abc
from linpyk.common import OpCode
from linpyk.graph.style import (
    NodeStyleOptions,
    default_data_node_style,
    default_control_node_style,
)
from typing import Any


class BaseNode(abc.ABC):
    """Base class for graph nodes. This class is abstract.

    Parameters
    ----------
    label (str | NOne): Node label or None. Default None.
    style (NodeStyleOptions | None): Node style configuration or None. Default None.

    """

    def __init__(
        self, label: str | None = None, style: NodeStyleOptions | None = None
    ) -> None:
        self.label = label
        self.style = style

    def __repr__(self) -> str:
        return str(self.label)


class DataNode(BaseNode):
    """Base class for data nodes i.e nodes corresponding to task I/Os.

    Parameters
    ----------
    label (str | NOne): Node label or None. Default None.
    values (Any): Initial values.
    style (NodeStyleOptions): Node style configuration or None. Default default_data_node_style.
    """

    def __init__(
        self,
        label: str | None = None,
        values: Any = None,
        style: NodeStyleOptions = default_data_node_style,
    ) -> None:
        super().__init__(label, style)
        self.result_id = None
        self.values = values


class ControlNode(BaseNode):
    """Base class for control nodes i.e nodes corresponding to tasks.

    Parameters
    ----------
    opcode (OpCode): The operation code designating the task referenced by the node.
    label (str | NOne): Node label or None. Default None.
    style (NodeStyleOptions): Node style configuration or None. Default default_control_node_style.
    **params: Additional keyword arguments to provide parameters to the task.
    """

    num_instances = 0

    def __init__(
        self,
        opcode: OpCode,
        label: str | None = None,
        style: NodeStyleOptions = default_control_node_style,
        **params,
    ) -> None:
        ControlNode.num_instances += 1
        super().__init__(f"{label} {ControlNode.num_instances}", style)
        self.opcode = opcode
        self.params = params


class TileNode(DataNode):
    """Class for data nodes referencing matrix tiles.

    Parameters
    ----------
    shape: Tile shape.
    label (str | NOne): Node label or None. Default None.
    values (Any): Initial values.
    style (NodeStyleOptions): Node style configuration or None. Default default_data_node_style.
    """

    def __init__(self, shape, label: str | None = None, values: Any = None) -> None:
        super().__init__(label, values=values)
        self.shape = shape

    def __repr__(self) -> str:
        if self.shape is not None:
            return f"{self.label} {self.shape} array"
        else:
            return f"TileNode {self.shape} array"
