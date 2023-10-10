"""
Worker execution workflow.
"""

import numpy as np
import logging

from armonik.worker import ClefLogger
from linpyk.common.dto import OpCode
from linpyk.common.payload import Payload
from linpyk.common.result import Result
from scipy.linalg import blas, lapack
from typing import Dict, List


class WorkerProcessingInstance:
    """
    Class implementing the interface with compute kernels. It manages the
    serialization and deserialization of data, the execution of
    computation using kernels and the handling of runtime errors.

    Parameters
    ----------
    payload : bytes
        A task payload.
    expected_output_ids : List[str]
        List of keys for expected outputs.
    data_dependencies : Dict[str, bytearray]
        Dictionnary containg the data dependencies as byte arrays given by
        their identification key.
    """

    def __init__(
        self,
        payload: bytes,
        expected_output_ids: List[str],
        data_dependencies: Dict[str, bytearray],
        logger: ClefLogger | None = None,
    ) -> None:
        deserialized_payload = Payload.deserialize(payload)
        self._opcode = OpCode(deserialized_payload.opcode)
        self._output_names = deserialized_payload.expected_output_names
        self._outputs = {}
        self._inputs = {}
        self._options = deserialized_payload.params
        self._logger = logger

        if deserialized_payload.dependency_names and data_dependencies:
            for name, uuid in deserialized_payload.dependency_names.items():
                self._inputs[name] = Result.deserialize(data_dependencies[uuid]).array

        if not (
            [*self._output_names.values()] in expected_output_ids
            or [*self._output_names.values()] == expected_output_ids
        ):
            raise ValueError(
                f"Invalid payload outputs ids {self._output_names.values()}"
            )

        # self._validate()

    def _validate(self):
        """
        Validates information received. In particular, it checks that
        the metadata provided  by the payload are consistent with each
        other and with the other data provided when the class is instantiated.
        """
        raise NotImplementedError()

    def process(self) -> None:
        """
        Run the requested compute kernel, manages results and error retrieval.
        """

        if self._logger is not None:
            self._logger.info(f"Performing operation with code {self._opcode}.")

        match self._opcode:
            case OpCode.DRGM:
                arr = None
                size = self._options["size"]
                idx = self._options["idx"]
                match self._options["gen"]:
                    case "eye":
                        if idx[0] == idx[1]:
                            arr = np.eye(size, dtype=np.float64)
                        else:
                            arr = np.zeros((size, size))
                    case "diag":
                        if idx[0] == idx[1]:
                            arr = np.diag(0.5 + 1.5*np.random.random(size))
                        else:
                            arr = np.zeros((size, size))
                    case _:
                        raise RuntimeError(f"Unhandled matrix generation {self._options['gen']}")
                self._outputs["a"] = Result(arr)
            case OpCode.POTRF:
                res, info = lapack.dpotrf(self._inputs["a"], **self._options)
                if info != 0:
                    raise RuntimeError(
                        f"Cholesky factorization of {self._inputs['a']} failed."
                    )
                self._outputs["l"] = Result(np.tril(res))
            case OpCode.TRSM:
                self._outputs["x"] = Result(
                    blas.dtrsm(
                        a=self._inputs["a"], b=self._inputs["b"], **self._options
                    )
                )
            case OpCode.SYRK:
                self._outputs["c"] = Result(
                    blas.dsyrk(
                        a=self._inputs["a"], c=self._inputs["c"], **self._options
                    )
                )
            case OpCode.GEMM:
                self._outputs["c"] = Result(
                    blas.dgemm(
                        a=self._inputs["a"],
                        b=self._inputs["b"],
                        c=self._inputs["c"],
                        **self._options,
                    )
                )
            case OpCode.ONES:
                self._outputs["o"] = Result(np.ones(**self._options))
            case _:
                raise RuntimeError(f"Unsupported opcode {self._opcode}.")

    def get_results(self) -> Dict[str, bytes]:
        """
        Return the results for the given computation.

        Returns
        ----------
        Dict[str, bytes]
            Dictionary containing each computed result key and the result data in bytes.
        """
        outputs = {}
        for name, result in self._outputs.items():
            if result:
                outputs[self._output_names[name]] = result.serialize()
        return outputs
