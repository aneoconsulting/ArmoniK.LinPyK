"""
Base exceptions and errors for ArmoniKLinAlg
"""


class ArmoniKLAException(Exception):
    """Base exception in ArmoniKlinpyk."""


class GraphIntegrityError(ArmoniKLAException):
    """Exception raised if an ExecGraph is not a directed acyclic bipartite graph."""


class GraphOpError(ArmoniKLAException):
    """Exception raised if a graph operation is not regular."""


class OperationError(ArmoniKLAException):
    """Raise when there is an error with a graph operation."""


class BackendError(ArmoniKLAException):
    """Exception raised when an error occurred in the backend."""
