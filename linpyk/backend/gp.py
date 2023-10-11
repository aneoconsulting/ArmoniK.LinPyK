"""
Basic backend to run code using any ArmoniK backend locally or remotely.
"""
import uuid
import grpc
import logging

from armonik.client import ArmoniKSubmitter, ArmoniKResult
from armonik.common import TaskOptions, TaskDefinition, ResultAvailability
from datetime import timedelta
from linpyk.backend.base import BaseBackend
from linpyk.common.result import Result
from linpyk.exceptions import BackendError
from typing import Dict, List, Union


class ArmoniKBackend(BaseBackend):
    """
    Basic backend to run program with an ArmoniK cluster locally or remotely.

    Examples
    ----------
    >>> backend = Backend("172.28.34.255:5001/", ["default"], TaskOptions(
            max_duration=timedelta(seconds=30),
            priority=1,
            max_retries=5,
            partition_id=["default"],
        )
    )
    """

    def __init__(
        self,
        endpoint: str,
        partitions: List[str],
        default_task_options: TaskOptions,
    ) -> None:
        super().__init__(endpoint, partitions, default_task_options)
        self._logger = logging.getLogger(__name__)
        self._channel = grpc.insecure_channel(self._endpoint).__enter__()
        self._client = ArmoniKSubmitter(self._channel)
        self._result_client = ArmoniKResult(self._channel)
        self._session_id = None
        self._submitted_tasks = []

    def __del__(self):
        # self.cancel_session()
        pass

    def create_session(self) -> str:
        self._session_id = self._client.create_session(
            default_task_options=self._default_task_options,
            partition_ids=self._partitions,
        )
        self._logger.info(f"Created session {self._session_id}")
        return self._session_id

    def resume_session(self, session_id: str) -> str:
        raise NotImplementedError

    def cancel_session(self) -> None:
        if self._session_id is not None:
            self._client.cancel_session(self._session_id)
            self._logger.warning(f"Session {self._session_id} has been closed")

    def submit_tasks(self, tasks: List[TaskDefinition], partition: str = "default") -> None:
        if self._session_id is not None:
            self._logger.info(f"Submitting tasks in session {self._session_id}")
            tasks, submission_errors = self._client.submit(self._session_id, tasks, TaskOptions(
                max_duration=timedelta(seconds=300),
                priority=1,
                max_retries=5,
                partition_id=partition
            ))
            for task in tasks:
                if self._logger.getEffectiveLevel() == logging.DEBUG:
                    self._logger.debug(f"Submitted task: {task}")
                else:
                    self._logger.info(f"Submitted task {task.id}")
            self._submitted_tasks.extend(tasks)

            for error in submission_errors:
                self._logger.critical(f"Submission error: {error}")
                raise BackendError(f"Submission error: {error}")

    def submit_results(self, results: Dict[int, Result]) -> None:
        for result_id, result in results.items():
            self._submit_result(result_id, result.serialize())
            self._logger.info(f"Submitted result {result_id}")

    def _submit_result(self, key: str, data: Union[bytes, bytearray]) -> None:
        """Send a result to ArmoniK.

        Parameters
        ----------
            key (str): Result key.
            data (Union[bytes, bytearray]): Result data.
        """
        from armonik.protogen.common.results_common_pb2 import UploadResultDataRequest

        configuration = self._client.get_service_configuration()

        def upload_result_stream():
            request = UploadResultDataRequest(
                id=UploadResultDataRequest.ResultIdentifier(
                    session_id=self._session_id, result_id=key
                )
            )
            yield request

            start = 0
            data_len = len(data)
            while start < data_len:
                chunksize = min(configuration.data_chunk_max_size, data_len - start)
                res = UploadResultDataRequest(
                    data_chunk=data[start : start + chunksize]
                )
                yield res
                start += chunksize

        self._result_client._client.UploadResultData(upload_result_stream())

    def request_output_id(self, num: int = 1) -> List[str]:
        if self._session_id is not None:
            self._logger.info(f"Creating {num} results.")
            result_ids = self._result_client.get_results_ids(self._session_id, [str(uuid.uuid4()) for _ in range(num)]).values()
            self._logger.info(f"Created {num} results.")
            return list(result_ids)

    def get_result(self, result_id: str) -> Union[Result, None]:
        self._logger.info(f"Retrieving result {result_id}...")
        if self._session_id is not None:
            if result_id is None:
                raise ValueError("'result_id' can't be None.")
            result = self._client.get_result(self._session_id, result_id=result_id)
            if result is None:
                self._logger.critical(f"Result {result_id} unexpectedly unavailable")
                raise BackendError(f"Result {result_id} unexpectedly unavailable.")
            return Result.deserialize(result)

    def wait_for_availability(self, result_id: str) -> Union[ResultAvailability, None]:
        if self._session_id is not None:
            self._logger.info(f"Waiting for result {result_id}...")
            return self._client.wait_for_availability(self._session_id, result_id)
