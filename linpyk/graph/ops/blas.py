"""
Graph operators to represent tasks implementing BLAS operations.
"""

from linpyk.common import OpCode
from linpyk.graph.nodes import ControlNode, TileNode
from linpyk.graph import graph


def tile_trsm(
    alpha: float,
    a: TileNode,
    b: TileNode,
    side: int = 0,
    lower: int = 0,
    trans_a: int = 0,
    diag: int = 0,
    overwrite_b: int = 0,
) -> TileNode:
    """Graph operator that adds the computational task corresponding to the DTRSM operation applied
    to a node representing a matrix tile.

    DTRSM  solves one of the matrix equations
        op( A ) * X = alpha * B,   or   X * op( A ) = alpha * B,
    where alpha is a scalar, X and B are m by n matrices, A is a unit, or
    non-unit,  upper or lower triangular matrix  and  op( A )  is one  of
        op( A ) = A   or   op( A ) = A**T.

    Parameters
    ----------
    alpha (float): The value of the alpha coefficient in the equation.
    a (TileNode): Graph node referencing a unit, or non-unit, upper or lower triangular matrix.
    b (TileNode): Graph node referencing a m by n matrix.
    side (int): Specify whether op( A ) appears on the left (side = 0) or right (side = 1) of X. Default 0.
    lower (int): Specify whether the matrix A is an upper (lower = 0) or lower (lower =1) triangular matrix.
        Default 0.
    trans_a (int): Specify the form of op( A ) to be used in the matrix multiplication. Set to 0
        op( A ) = A and set to 1, op( A ) = A**T. Default 0.
    diag (int): Specify whether A is unit triangular (diag = 1) or not (diag = 0). Default 0.
    overwrite_b (int): The matrix X is overwritten on B if set to 1. Default 0.

    Return
    ----------
    TileNode : Node representing the output of the computation.

    """

    x = TileNode(b.shape)
    control = ControlNode(
        OpCode.TRSM,
        "TRSM",
        alpha=alpha,
        side=side,
        lower=lower,
        trans_a=trans_a,
        diag=diag,
        overwrite_b=overwrite_b,
    )
    graph.current_graph.add_edges_from(
        [
            (a, control, {"input_name": "a"}),
            (b, control, {"input_name": "b"}),
            (control, x, {"output_name": "x"}),
        ]
    )
    return x


def tile_syrk(
    alpha: float,
    a: TileNode,
    c: TileNode,
    beta: float = 0.0,
    trans: int = 0,
    lower: int = 0,
    overwrite_c: int = 0,
) -> TileNode:
    """Graph operator that adds the computational task corresponding to the DSYRK operation applied
    to a node representing a matrix tile.

    DSYRK performs one of the symmetric rank k operations:
        C := alpha*A*A**T + beta*C, or C := alpha*A**T*A + beta*C,
    where alpha and beta  are scalars, C is an n by n symmetric matrix and A is
    an n by k matrix in the first case and a k by n in the second case.

    Parameters
    ----------
    alpha (float): The value of the alpha coefficient in the equation.
    a (TileNode): Graph node referencing a n by n symetric matrix.
    c (TileNode): Graph node referencing a lower triangular or non-triangular n by n matrix.
    beta (float): The value of the beta coefficient in the equation. Default 0.0.
    trans (int): Specify if the operation to be performed corresponds to first case (trans = 0) or the
        second case (trans = 1). Default 0.
    lower (int): Specify whether the upper (lower = 0) or lower (lower = 1) triangular part of the matrix
        C is to be referenced. Default 0.
    overwrite_c: The matrix C is overwritten if set to 1. Default 0.

    Return
    ----------
    TileNode : Node representing the output of the computation.

    """

    x = TileNode(a.shape)
    control = ControlNode(
        OpCode.SYRK,
        "SYRK",
        alpha=alpha,
        beta=beta,
        trans=trans,
        lower=lower,
        overwrite_c=overwrite_c,
    )
    graph.current_graph.add_edges_from(
        [
            (a, control, {"input_name": "a"}),
            (c, control, {"input_name": "c"}),
            (control, x, {"output_name": "c"}),
        ]
    )
    return x


def tile_gemm(
    alpha: float,
    a: TileNode,
    b: TileNode,
    beta: float = 0.0,
    c: TileNode | None = None,
    trans_a: int = 0,
    trans_b: int = 0,
    overwrite_c: int = 0,
) -> TileNode:
    """Graph operator that adds the computational task corresponding to the DGEMM operation applied
    to a node representing a matrix tile.

    DGEMM  performs one of the matrix-matrix operations
        C := alpha * op( A ) * op( B ) + beta * C,
    where  op( X ) is one of
        op( X ) = X   or   op( X ) = X**T,
    alpha and beta are scalars, and A, B and C are matrices, with op( A )
    an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix.

    Parameters
    ----------
    alpha (float): The alpha coefficient in the operation.
    a (TileNode): Graph node referencing a m by k matrix.
    b (TileNode): Graph node referencing a k by n matrix.
    beta (float): The bet coefficient in the operation. Default 0.0.
    c (TileNode | None): Optional graph node referencing a m by n matrix. If beta is 0 this matrix is
        not required and can be replaced by a None. Default None.
    trans_a (int): Specify the form of op( A ) to be used in the matrix multiplication. Op( A ) = A if
        set to 0 or op( A ) = A**T is set to 1. Default 0.
    trans_b (int): Specify the form of op( B ) to be used in the matrix multiplication. Op( B ) = B if
        set to 0 or op( B ) = B**T is set to 1. Default 0.
    overwrite_c (int): The matrix C is overwritten if set to 1. Default 0.

    Return
    ----------
    TileNode : Node representing the output of the computation.

    """

    x = TileNode(a.shape)
    control = ControlNode(
        OpCode.GEMM,
        "GEMM",
        alpha=alpha,
        beta=beta,
        trans_a=trans_a,
        trans_b=trans_b,
        overwrite_c=overwrite_c,
    )
    graph.current_graph.add_edges_from(
        [
            (a, control, {"input_name": "a"}),
            (b, control, {"input_name": "b"}),
            (control, x, {"output_name": "c"}),
        ]
    )
    if c is not None:
        graph.current_graph.add_edge(c, control, input_name="c")
    return x
