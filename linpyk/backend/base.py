"""
Base class for backend.
"""

import abc
import networkx as nx

from armonik.common import TaskOptions, TaskDefinition, ResultAvailability
from contextlib import contextmanager
from linpyk.common import Payload, Result
from linpyk.graph import ArmoniKGraph, ControlNode, DataNode, DTArray
from typing import Dict, List, Union


class BaseBackend(abc.ABC):
    """
    Base class for backend. A backend is an object abstracting the computing infrastructure
    where tasks are run.

    Parameters
    ----------
    endpoint : str
        Endpoint of the control plane of the ArmoniK cluster used as the computing infrastructure.
    partitions : List[str]
        Cluster partitions where to run tasks.
    default_task_options : TaskOptions
        Default tasks options for the session.

    """

    def __init__(
        self, endpoint: str, partitions: List[str], default_task_options: TaskOptions
    ) -> None:
        self._endpoint = endpoint
        self._partitions = partitions
        self._default_task_options = default_task_options

    @contextmanager
    def new_session(self):
        self.create_session()
        yield SessionContext(self)
        self.cancel_session()

    @abc.abstractmethod
    def create_session(self) -> str:
        """
        Create a new session.
        """
        pass

    @abc.abstractmethod
    def resume_session(self, session_id: str) -> str:
        """
        Resume an existing session.
        """
        pass

    @abc.abstractmethod
    def cancel_session(self) -> None:
        """
        Cancel a running session.
        """
        pass

    @abc.abstractmethod
    def submit_tasks(self, tasks: List[TaskDefinition]) -> None:
        """
        Submit a list of tasks to the ArmoniK cluster.

        Parameters
        ----------
        tasks : List[TaskDefinition]
            Definitions of the tasks to run.
        """
        pass

    @abc.abstractmethod
    def submit_results(self, results: Dict[int, Result]) -> None:
        """
        Submit a list of results to the ArmoniK cluster. This method is intended to be used
        to send tasks input data that are not outputs of other tasks.

        Parameters
        ----------
        results : Dict[int, Results]
            A dictionary which keys are results uuids and values are results.
        """
        pass

    @abc.abstractmethod
    def request_output_id(self):
        """
        Request for a result unique id.
        """
        pass

    @abc.abstractmethod
    def get_result(self, result_id: str) -> Union[Result, None]:
        """
        Retrieve a result if possible.

        Parameters
        ----------
        result_id : str
            Result unique ID.

        Return
        ----------
        Result | None
            An object Result containing the output data if the result exists. None otherwise.
        """
        pass

    @abc.abstractmethod
    def wait_for_availability(self, result_id: str) -> Union[ResultAvailability, None]:
        """
        Blocks until the result is available or is in error

        Parameters
        ----------
        result_id : str
            The result unique ID.

        Returns
        ----------
        ResultAvailability | None
            None if the wait was cancelled unexpectedly, otherwise a
            ResultAvailability with potential errors.
        """
        pass


class SessionContext:
    def __init__(self, backend: BaseBackend) -> None:
        self._backend = backend

    def collect(self):
        raise NotImplementedError()

    def run(self, exec_graph: ArmoniKGraph) -> None:
        self._backend.submit_results(self._pre_compile(exec_graph))
        self._backend.submit_tasks(SessionContext._compile(exec_graph))

    def _pre_compile(self, exec_graph: ArmoniKGraph) -> Dict[int, Result]:
        """Perform some operations required for graph compilation:
            - associate each data node with a result id.
            - retrieve initial results to be upload to the cluster.

        Parameters
        ----------
        exec_graph: ArmoniKGraph
            The graph to be compiled.

        Return
        ----------
        Dict[int, Result] : The results to be uploaded before the graph execution.
        """
        for node in exec_graph.nodes:
            if isinstance(node, DataNode):
                node.result_id = self._backend.request_output_id()

        results = {}
        for node in exec_graph.nodes:
            if (
                isinstance(node, DataNode)
                and exec_graph.in_degree(node) == 0
                and exec_graph.out_degree(node) > 0
            ):
                results[node.result_id] = Result(node.values)
        return results

    @classmethod
    def _compile(cls, exec_graph: ArmoniKGraph) -> List[TaskDefinition]:
        """Compile the execution graph i.e. returns the list of
        task definitions corresponding the graph.

        Parameters
        ----------
        exec_graph : ArmoniKGraph
            The execution graph to compile.

        Returns
        ----------
        List[TaskDefinition]
            The list of task corresponding to the graph.
        """
        tasks = []
        for node in nx.topological_sort(exec_graph):
            if isinstance(node, ControlNode):
                expected_output_names = {
                    exec_graph.edges[node, successor][
                        "output_name"
                    ]: successor.result_id
                    for successor in exec_graph.successors(node)
                }
                dependency_names = {
                    exec_graph.edges[predecessor, node][
                        "input_name"
                    ]: predecessor.result_id
                    for predecessor in exec_graph.predecessors(node)
                }
                tasks.append(
                    TaskDefinition(
                        Payload(
                            node.opcode,
                            expected_output_names,
                            dependency_names,
                            node.params,
                        ).serialize(),
                        expected_output_ids=list(expected_output_names.values()),
                        data_dependencies=list(dependency_names.values()),
                    )
                )
        return tasks

    def get_data_node(self, data_node: DataNode) -> None:
        reply = self._backend.wait_for_availability(data_node.result_id)
        if reply.is_available():
            result = self._backend.get_result(data_node.result_id)
        else:
            result = None
        if result is None:
            raise RuntimeError(f"{data_node.label} is None.")
        else:
            data_node.values = result.array

    def get(self, data: Union[DataNode, DTArray]) -> None:
        if isinstance(data, DataNode):
            self.get_data_node(data)
        elif isinstance(data, DTArray):
            for tile in data.tiles():
                self.get_data_node(tile)
