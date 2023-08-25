"""
Worker to support the linear algebra library.
"""

import grpc
import logging
import os

from armonik.common import Output, HealthCheckStatus
from armonik.worker import ArmoniKWorker, TaskHandler, ClefLogger
from linpyk.common import WorkerProcessingInstance


ClefLogger.setup_logging(logging.INFO)


def health_check() -> HealthCheckStatus:
    """
    Checks the worker's health status.

    Returns
    ----------
    HealthCheckStatus
        The health status.
    """
    return HealthCheckStatus.SERVING


def processor(task_handler: TaskHandler) -> Output:
    """
    Implements the main logic of the worker.

    Parameters
    ----------
    task_handler: TaskHandler
        Wrapper over scheduling agent API.

    Returns
    ----------
    Output
        An empty Output object if all went well, or containing a string describing the
        error otherwise.
    """

    logger = ClefLogger.getLogger("ArmoniKWorker")

    instance = WorkerProcessingInstance(
        task_handler.payload,
        task_handler.expected_results,
        task_handler.data_dependencies,
        logger,
    )

    try:
        logger.info("Worker is running...")
        instance.process()
    except RuntimeError as error:
        logger.error(f"Fatal error occured: {error}")
        return Output(str(error))

    for result_id, result in instance.get_results().items():
        task_handler.send_result(result_id, result)

    logger.info("Work completed successfully.")

    return Output()


def main() -> None:
    """
    Worker entry point.

    Returns
    ----------
        None.
    """

    # Create Seq compatible logger
    logger = ClefLogger.getLogger("ArmoniKWorker")

    # Define agent agent-worker communication endpoints
    worker_scheme = (
        "unix://"
        if os.getenv("ComputePlane__WorkerChannel__SocketType", "unixdomainsocket")
        == "unixdomainsocket"
        else "http://"
    )
    agent_scheme = (
        "unix://"
        if os.getenv("ComputePlane__AgentChannel__SocketType", "unixdomainsocket")
        == "unixdomainsocket"
        else "http://"
    )
    worker_endpoint = worker_scheme + os.getenv(
        "ComputePlane__WorkerChannel__Address", "/cache/armonik_worker.sock"
    )
    agent_endpoint = agent_scheme + os.getenv(
        "ComputePlane__AgentChannel__Address", "/cache/armonik_agent.sock"
    )

    # Start worker
    logger.info("Worker just started!")
    with grpc.insecure_channel(agent_endpoint) as agent_channel:
        worker = ArmoniKWorker(agent_channel, processor, health_check, logger=logger)
        logger.info("Worker connected")
        worker.start(worker_endpoint)


if __name__ == "__main__":
    main()
