"""
Payload class to represent task payloads.
"""

import json

from dataclasses import dataclass
from linpyk.common import OpCode
from typing import Any, Dict, Optional


@dataclass
class Payload:
    """
    Provides an object structure to represent task payloads and facilitates
    serialization and deserialization of payloads in binary.

    Parameters
    ----------
    opcode : OpCode
        Indicate which operation needs to be performed.
    expected_output_names : Dict[str, str]
        Dictionary associating the unique IDs of results with comprehensible names.
    dependency_names (optional) : Dict[str, str]
        Dictionary associating the unique IDs of data dependencies with comprehensible names.
    params (optional) : Dict[str, Any]
        Additional parameters.
    """

    opcode: OpCode
    expected_output_names: Dict[str, str]
    dependency_names: Optional[Dict[str, str]] = None
    params: Optional[Dict[str, Any]] = None

    def serialize(self) -> bytes:
        """
        Serializes the payload. Converts the attributes to json and return a byte array.

        Returns
        ----------
        bytes
            Serialized payload compatible with ArmoniK.
        """
        return json.dumps(
            {
                "opcode": self.opcode,
                "params": self.params,
                "expected_output_names": self.expected_output_names,
                "dependency_names": self.dependency_names,
            }
        ).encode("utf-8")

    @classmethod
    def deserialize(cls, payload: bytes) -> "Payload":
        """
        Create a payload instance from the payload bytes received from ArmoniK.

        Parameters
        ----------
        payload : bytes
            Raw ArmoniK Payload.

        Returns
        ----------
        Payload
            Payload object.
        """
        return cls(**json.loads(payload.decode("utf-8")))
