"""
Provide matrix random generators.
"""

import numpy as np
import scipy as sp
from typing import Tuple


generator = np.random.mtrand._rand


def make_spd_matrix(n_dim: int) -> np.ndarray:
    """Generate a random symmetric, positive-definite matrix.

    Parameters
    ----------
    n_dim : int
        The matrix dimension.

    Returns
    -------
    X : ndarray of shape (n_dim, n_dim)
        The random symmetric, positive-definite matrix.
    """

    A = generator.uniform(size=(n_dim, n_dim))
    U, _, Vt = sp.linalg.svd(np.dot(A.T, A), check_finite=False)
    X = np.dot(np.dot(U, 1.0 + np.diag(generator.uniform(size=n_dim))), Vt)

    return X.astype("float64")


def make_spd_system(n_dim: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a random symmetric, positive-definite matrix A, a vector b and the solution x
    of the linear system Ax = b.

    Parameters
    ----------
    n_dim : int
        The matrix dimension.

    Returns
    -------
    A, X, b : Tuple[ndarray, ndarray, ndarray]
        The random symmetric, positive-definite matrix, the solution fo the system and the random vector.
    """
    a = make_spd_matrix(n_dim)
    x = generator.uniform(size=(n_dim, 1))
    b = np.dot(a, x)

    return a.astype("float64"), x.astype("float64"), b.astype("float64")


def make_triangular_system(n_dim, lower: bool = False, trans_a: bool = False):
    """Generate a random triangular matrix A and an associated matrix B such that there is
    a matrix X such that op( A ) * X = B where op( A ) = A or op( A ) = A**T.

    Parameters
    ----------
    n_dim : int
        The matrix dimension.
    lower : bool
        Compute a random lower triangular matrix.
    trans_a : bool
        Compute B such that B is the solution of A**T * X = B.

    Returns
    -------
    A, B : ndarray of shape (n_dim, n_dim)
        The random triangular matrix and the associated matrix.
    """

    A = generator.uniform(size=(n_dim, n_dim))
    X = generator.uniform(size=(n_dim, 1))

    if lower:
        A = np.tril(A)
    else:
        A = np.triu(A)

    if trans_a:
        B = np.dot(A.T, X)
    else:
        B = np.dot(A, X)

    return A.astype("float64"), X.astype("float64"), B.astype("float64")
