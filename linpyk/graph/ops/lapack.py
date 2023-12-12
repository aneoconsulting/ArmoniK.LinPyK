"""
Graph operators to represent tasks implementing LAPACK operations.
"""

from linpyk.common import OpCode
from linpyk.graph.nodes import ControlNode, TileNode
from linpyk.graph import graph


def tile_potrf(
    a: TileNode, lower: int = 0, clean: int = 0, overwrite_a: int = 1
) -> TileNode:
    """Graph operator that adds the computational task corresponding to the DPOTRF operation applied
    to a node representing a matrix tile.

    DPOTRF computes the Cholesky factorization of a real symmetric positive definite matrix A.
    The factorization has the form
        A = U**T * U,  if lower = 0, or
        A = L * L**T,  if lower = 1,
    where U is an upper triangular matrix and L is lower triangular.

    Parameters
    ----------
    a (TileNode): Graph node referencing a real symmetric positive definite matrix.
    lower (int): Specify whether the computed matrix is upper triangular (lower = 0) or lower triangular (lower = 1).
                 Default 0.
    clean (int): Whether to check if there are infinite values.
                 Default 0.
    overwrite_a (int): The matrix U or L is overwritten on A if set to 1.
                       Default 1.

    Return
    ----------
    TileNode : Node representing the output of the computation.

    """

    control = ControlNode(
        OpCode.POTRF, "POTRF", lower=lower, clean=clean, overwrite_a=overwrite_a
    )
    l = TileNode(a.shape)
    graph.current_graph.add_edges_from(
        [(a, control, {"input_name": "a"}), (control, l, {"output_name": "l"})]
    )
    return l
