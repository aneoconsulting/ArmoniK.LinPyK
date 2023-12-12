"""
Define opcodes for tasks. It serves as an interface between clients and workers.
"""

from enum import auto, IntEnum, unique


@unique
class OpCode(IntEnum):
    """
    Enumerates and encodes all operations that can be performed by a backend.

    Members
    ----------
    DRGM : int
        Computes a random squared matrix.
    DPOTRF : int
        Computes the Cholesky factorization of a real symmetric positive definite
        matrix A. The factorization has the form A = L * L**T.
    DTRSM : int
        Solves one of the following matrix equations
        op(A)*X = alpha*B,   or   X*op(A) = alpha*B, where alpha is a scalar,
        X and B are n by n matrices, A is a unit, or non-unit, upper or lower
        triangular matrix  and  op(A) is one of : op(A) = A   or   op(A) = A**T.
    DSYRK : int
        Performs one of the symmetric rank k operations
        C := alpha*A*A**T + beta*C, or C := alpha*A**T*A + beta*C,
        where alpha and beta  are scalars, C is an n by n symmetric matrix
        and  A  is an  n by n matrix.
    DGEMM : int
        Performs one of the matrix-matrix operations
        C := alpha*op(A)*op(B) + beta*C, where  op(X) is one of
        op(X) = X   or   op(X) = X**T,
        alpha and beta are scalars, and A, B and C are matrices, with op(A)
        an n by n matrix,  op( B )  a  n by n matrix and  C an n by n matrix.

    """

    DRGM = auto()
    POTRF = auto()
    TRSM = auto()
    SYRK = auto()
    GEMM = auto()
    ONES = auto()
