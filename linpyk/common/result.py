"""
Result class to represent task result.
"""

import pickle
import numpy as np

from dataclasses import dataclass


@dataclass
class Result:
    """
    Provides an object structure to represent task results and facilitates
    serialization and deserialization of results in binary.

    Parameters
    ----------
    array : numpy.ndarray
        Any N-dimensional numpy array.

    """

    array: np.ndarray

    def serialize(self) -> bytes:
        """
        Serializes the result. Converts the array to a byte array.

        Parameters
        ----------
        bytes
            Serialized result compatible with ArmoniK.
        """
        return pickle.dumps(self.array)

    @classmethod
    def deserialize(cls, payload: bytes) -> "Result":
        """
        Create a Result instance from the data dependency bytes received from ArmoniK.

        Parameters
        ----------
        payload : Raw ArmoniK Result.

        Returns
        ----------
        Result
            Result object.
        """
        return cls(pickle.loads(payload))
