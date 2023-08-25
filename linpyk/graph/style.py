"""
Nodes style configurations.
"""

from dataclasses import dataclass
from enum import Enum


class Shapes(Enum):
    CIRCLE = "circle"
    SQUARE = "square"


@dataclass(frozen=True)
class NodeStyleOptions:
    shape: str
    color: str


default_data_node_style = NodeStyleOptions("square", "green")

default_control_node_style = NodeStyleOptions("circle", "red")
