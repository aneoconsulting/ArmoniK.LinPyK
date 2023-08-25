"""
Base class for ArmoniK task graphs representation.
"""

import networkx as nx
from linpyk.exceptions import GraphIntegrityError
from typing import List


current_graph = None


class ArmoniKGraph(nx.DiGraph):
    """Base class for ArmoniK-compatible task graphs."""

    def __enter__(self):
        """Sets the instance on which the method is called as the working graph. It means that all the graph
        operators will implicitly be applied on this instance."""
        global current_graph
        current_graph = self

    def __exit__(self, exc_type, exc_value, exc_tb):
        """Reset the working graph."""
        global current_graph
        current_graph = None

    def _check_integrity(self):
        """
        Check that the graph is directed, acyclic and bipartite.

        Raises
        ------
        GraphIntegrityError
            If the graph is not directed acyclic and bipartite.
        """
        if not nx.is_directed_acyclic_graph(self):
            raise GraphIntegrityError("The graph is not directed acyclic.")

        for rn, ln in self.edges:
            if type(rn) == type(ln):
                raise GraphIntegrityError(
                    f"The graph is not bipartite: {rn} and {ln} are of the same type."
                )

    def data_successors(self, n) -> List:
        """
        Returns an iterator over the data successor nodes of n.

        Parameters
        ----------
        n : DataNode
           A data node in the graph

        Raises
        ------
        NetworkXError
           If n is not in the graph.
        ValueError
            If n is not a data node.

        Return
        --------
        AdjencyView
            An iterator over the data successor nodes of n
        """

        from linpyk.graph import DataNode

        if not isinstance(n, DataNode):
            raise ValueError(f"The node {n} is not of type 'DataNode'.")

        data_succ = []
        for control_succ in super().successors(n):
            data_succ.append(super().successors(control_succ))
        return data_succ

    def savefig(self, fname: str):
        """Save an image of the task graph.

        Parameters
        ----------
        fname (str): File name for the image.
        """

        for node in self.nodes:
            self.nodes[node]["label"] = node.label
            for key, value in node.style.__dict__.items():
                self.nodes[node][key] = value
        gv_graph = nx.nx_agraph.to_agraph(self)
        gv_graph.layout(prog="dot")
        gv_graph.draw(fname)
