"""
Classes for tiled arrays representation and handling in task graphs
"""

import numpy as np
from abc import ABC
from linpyk.graph import graph
from linpyk.graph.nodes import TileNode
from typing import Callable, Tuple, Generator, Union


ShapeLike = Union[int, Tuple[int, ...]]


class DTArray(ABC):
    """Class for tiled arrays representation and handling in task graphs.
    DTArray stands for Distributed Tiled Array. This class allows intuitive implementation of linear algebra algorithm.

    It is an abstraction over numpy arrays. It is intended to be used the same way.

    This class is abstract.

    Parameters
    ----------
    shape (ShapeLike): The shape of the array.
    label (str): The name of the array.
    """

    def __init__(self, shape: ShapeLike, label: str):
        self._tiles = np.empty(shape=shape, dtype=object)
        self._label = label

    @property
    def shape(self):
        """Returns the shape of the array."""
        return self._tiles.shape

    def tiles(self) -> Generator[TileNode, None, None]:
        """Iterate over the nodes referenced by the array."""
        for node in self._tiles.flat:
            if node is not None:
                yield node

    def _name_block(self, index: ShapeLike) -> str:
        """Generate a name for a tile of the matrix.

        Parameters
        ----------
        index: Tile index.

        Return
        ----------
        str : The generated name.
        """
        return f"{self._label}[{str(index).replace('(', '').replace(')', '')}]"

    def __getitem__(self, index: ShapeLike) -> TileNode:
        """Returns a given tile of the matrix.

        Parameters
        ----------
        index: Tile index.

        Return
        ----------
        TileNode : The TileNode indexed by 'index' in the array.
        """
        return self._tiles[index]

    def __setitem__(self, index: ShapeLike, tile: TileNode) -> None:
        """Set or reset a tile referenced by the array.

        Parameters
        ----------
        index: Tile index.
        tile (TileNode): The TileNode to be referenced with 'index'.
        """

        # Change the name of the node so it can be seen as a tile of the array
        tile.label = self._name_block(index)

        # When an item is added to a DTArray the node is added to the working graph
        graph.current_graph.add_node(tile)

        self._tiles[index] = tile

    @classmethod
    def from_numpy(cls, label: str, array: np.ndarray, tile_size_map: Callable[[ShapeLike], ShapeLike]):


        for size in array.shape:
            if size % tile_size != 0:
                raise ValueError()
        shape = tuple([size // tile_size for size in array.shape])
        dt_array = DTArray(shape, label)
        dt_array.set_from_array(array, lambda _: (tile_size, tile_size))

    def set_from_array(
        self, array: np.ndarray, tile_size_map: Callable[[int, int], Tuple[int, int]]
    ):
        """Initializes a DTArray from a numpy array.
        This method only works for 2D-arrays.

        Parameters
        ----------
        array (numpy.array): Array to be tiled and converted into a DTArray.
        tile_size_map (Callable[[int, int], Tuple[int, int]]): A function mapping indices to the shapes of the
        corresponding tiles.
        """

        if len(array.shape) > 2:
            raise NotImplementedError()
        for index, _ in np.ndenumerate(self._tiles):
            tile_shape = tile_size_map(index)
            i, j = index[0], index[1]
            bi, bj = tile_shape[0], tile_shape[1]
            self[index] = TileNode(
                tile_shape, values=array[i * bi : (i + 1) * bi, j * bj : (j + 1) * bj]
            )

    def to_numpy(self, triangular: bool = False) -> np.ndarray:
        """Converts the DTArray into a numpy array. This requires that all tiles correspond to results that have been
        correctly computed and retrieved from the cluster.

        If a tile has not been computed (the index correspond to a None or the TileNode values are empty) then the
        function throw an exception.

        Parameters
        ----------
        triangular (bool): Specify whether the matrix is triangular. If the matrix is triangular a part
        of the tiles are empty but correspond to tiles of zeros so the conversion can be performed anyway.

        Return
        ----------
        numpy.array : The numpy array.

        Exceptions
        ----------
        ValueError : Throw when the conversion can't be performed because an item of the array is not set.

        """
        blocks = np.empty(self._tiles.shape, dtype=object)
        for index, tile in np.ndenumerate(self._tiles):
            if isinstance(tile, TileNode):
                blocks[index] = tile.values
            else:
                if triangular and isinstance(self._tiles[index[::-1]], TileNode):
                    blocks[index] = np.zeros(self._tiles[index[::-1]].shape[::-1])
                else:
                    raise ValueError(
                        f"Block of index {index} is not of type 'TileNode'. Only fully set "
                        f"arrays can be converted to numpy array."
                    )
        return np.block(blocks.tolist())

    def _transpose(self):
        self._tiles = self._tiles.T
        for index, tile in np.ndenumerate(self._tiles):
            if tile is not None:
                self._tiles[index].values = tile.values.T


class Empty(DTArray):
    """A class for empty DTArrays."""

    def __init__(self, shape, label):
        super().__init__(shape, label)


class Pascal(DTArray):
    """
    DTArray initialized as a lower Pascal matrix of size n x n.
    Each element of the matrix contains an integer representing the corresponding value in the Pascal matrix.

    Parameters
    ----------
    n (int): The size of the Pascal matrix to create.
    b (int): The size of the tiles.
    lower (bool): Whether to create a lower or upper triangular matrix. Default False.

    Example
    ----------
    >>> pascal = Pascal(6, 2):
    >>> pascal.to_numpy()
        [[ 1.  0.  0.  0.  0.  0.]
        [ 1.  1.  0.  0.  0.  0.]
        [ 1.  2.  1.  0.  0.  0.]
        [ 1.  3.  3.  1.  0.  0.]
        [ 1.  4.  6.  4.  1.  0.]
        [ 1.  5. 10. 10.  5.  1.]]

    """

    def __init__(self, label, n: int, b: int, lower: bool = True):
        if n % b != 0:
            raise ValueError("Tiles size must fit the size of the matrix.")

        super().__init__((n // b, n // b), label)

        pascal = np.zeros((n, n), dtype=int)
        pascal[0, 0] = 1

        for i in range(1, n):
            for j in range(i + 1):
                if j == 0:
                    pascal[i, j] = pascal[i - 1, j]
                elif j == i:
                    pascal[i, j] = pascal[i - 1, j - 1]
                else:
                    pascal[i, j] = pascal[i - 1, j] + pascal[i - 1, j - 1]

        pascal.astype(float)

        self.set_from_array(pascal, lambda index: (b, b))

        if not lower:
            self._transpose()


class Hilbert(DTArray):
    """
    DTArray initialized as a Hilbert matrix of size N x N,

    Parameters
    ----------
    n (int): The size of the Pascal matrix to create.
    b (int): The size of the tiles.

    Example
    ----------
    >>> hilbert = Hilbert(6, 2):
    >>> hilbert.to_numpy()
       [[2.         0.5        0.33333333 0.25       0.2        0.16666667]
        [0.5        1.33333333 0.25       0.2        0.16666667 0.14285714]
        [0.33333333 0.25       1.2        0.16666667 0.14285714 0.125     ]
        [0.25       0.2        0.16666667 1.14285714 0.125      0.11111111]
        [0.2        0.16666667 0.14285714 0.125      1.11111111 0.1       ]
        [0.16666667 0.14285714 0.125      0.11111111 0.1        1.09090909]]
    """

    def __init__(self, label, n: int, b: int):
        if n % b != 0:
            raise ValueError("Tiles size must fit the size of the matrix.")
        nb = n // b

        super().__init__((nb, nb), label)

        hilbert = np.array(np.eye(n))
        size_p = 1.0
        for m in range(nb):
            for n in range(nb):
                for mm in range(b):
                    for nn in range(b):
                        row = m * b + mm
                        col = n * b + nn
                        hilbert[row, col] = 1.0 / (1.0 + (n * b + mm) + (m * b + nn))
                        if n == m and mm == nn:
                            # Improvement of numerical stability: Adding a small number size_p
                            # to these diagonal elements can improve numerical stability when computing the matrix inverse.
                            # and reduce sensitivity to round-off errors.
                            hilbert[row, col] += 1.0 * size_p
        self.set_from_array(hilbert, lambda _: (b, b))
