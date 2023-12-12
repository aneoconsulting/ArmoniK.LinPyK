import logging
import numpy as np
import os
import time

from datetime import timedelta
from armonik.common import TaskOptions
from linpyk.algorithms import cholesky
from linpyk.backend import ArmoniKBackend, DummyBackend
from linpyk.graph import ArmoniKGraph, hilbert


start = time.time()

endpoint = os.environ.get("ARMONIK_BACKEND", "192.168.6.61:5001")
os.environ["LINPYK_LOGGING"] = "INFO"

backend = ArmoniKBackend(
    endpoint,
    ["default"],
    TaskOptions(
        max_duration=timedelta(seconds=300),
        priority=1,
        max_retries=5,
        partition_id="default",
    ),
)
backend = DummyBackend()

g = ArmoniKGraph()

size = 6
block_size = 2

with g:
    a = hilbert("A", size, block_size)
    l = cholesky(a)

g.savefig("./graph.png")

with backend.new_session() as sc:
    sc.run(g)
    sc.get(l)

numpy_a = a.to_numpy()
numpy_l = l.to_numpy(triangular=True)

print(
    f"Max coefficient difference: {np.max((np.matmul(numpy_l, numpy_l.T) - numpy_a).flat)}"
)

end = time.time()

print(f"Total execution time: {end - start}s.")
