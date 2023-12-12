"""
Graph operators for generating matrices remotely on the cluster.
"""

from linpyk.common import OpCode
from linpyk.graph.array import DTArray
from linpyk.graph.nodes import ControlNode, TileNode
from linpyk.graph import graph


def remote_dtarray_generator(n_dim: int, tile_size: int, generate: str) -> DTArray:
    """
    Graph operator to generate remotely a matrix of a given type.

    Parameters
    ----------
    n_dim : int
        The number of tiles of the squared array.
    tile_size : int
        The dimension of a squared tile.
    generate : str
        The type of the array (identity, spd).

    Returns
    ----------
    a : DTArray
        A DTArray object representing the generated matrix.
    """

    a = DTArray((n_dim, n_dim), "A")
    for i in range(n_dim):
        for j in range(n_dim):
            a[i, j] = TileNode((tile_size, tile_size))

    controls = {idx: ControlNode(
        OpCode.DRGM,
        "DRGM",
        gen=generate,
        size=tile_size,
        idx=idx
    ) for idx, _ in a.tiles(index=True)}

    graph.current_graph.add_edges_from([(control, a[idx], {"output_name": "a"}) for idx, control in controls.items()])

    return a
