"""
Dummy backend to run code without an ArmoniK backend.
"""

import uuid

from armonik.common import TaskOptions, TaskDefinition, ResultAvailability
from datetime import timedelta
from linpyk.backend.base import BaseBackend
from linpyk.common import Result, WorkerProcessingInstance
from linpyk.exceptions import BackendError
from typing import Dict, List, Union


class DummyBackend(BaseBackend):
    """ "
    Dummy backend to run program without an ArmoniK cluster.

    This backend can be instantiated without arguments or by passing suitable arguments.
    The second option minimizes code modification when switching this backend with a real one.

    Example
    ----------
    >>> backend = NoBackend()
    """

    # pylint: disable=W0102
    def __init__(
        self,
        endpoint: str = "",
        partitions: List[str] = [],
        default_task_options: TaskOptions = TaskOptions(
            max_duration=timedelta(seconds=1),
            priority=1,
            max_retries=1,
            partition_id=[],
        ),
    ) -> None:
        super().__init__(endpoint, partitions, default_task_options)
        self._results = dict()

    def create_session(self) -> str:
        return "No session: running without an ArmoniK cluster."

    def resume_session(self, session_id: str) -> str:
        pass

    def cancel_session(self) -> None:
        pass

    def submit_tasks(self, tasks: List[TaskDefinition], partition: str = "Default") -> None:
        for task in tasks:
            task_data_dependencies = {}
            if task.data_dependencies:
                for data_dependency in task.data_dependencies:
                    if self._results[data_dependency] is not None:
                        task_data_dependencies[data_dependency] = self._results[
                            data_dependency
                        ]
                    else:
                        raise BackendError(
                            "A task with unresolved dependencies has been encountered. Perhaps the tasks were not submitted in topological order."
                        )
            instance = WorkerProcessingInstance(
                task.payload, task.expected_output_ids, task_data_dependencies
            )
            instance.process()
            for result_id, result in instance.get_results().items():
                self._results[result_id] = result

    def submit_results(self, results: Dict[int, Result]) -> None:
        for result_id, result in results.items():
            try:
                self._results[result_id] = result.serialize()
            except KeyError:
                raise BackendError("Result submitted before being created.")

    def request_output_id(self):
        result_id = str(uuid.uuid4())
        self._results[result_id] = None
        return result_id

    def get_result(self, result_id: str) -> Union[Result, None]:
        return Result.deserialize(self._results[result_id])

    def wait_for_availability(self, result_id: str) -> Union[ResultAvailability, None]:
        return ResultAvailability()
