import argparse
import numpy as np

from armonik.common import TaskOptions
from datetime import timedelta
from linpyk.algorithms import cholesky
from linpyk.backend import ArmoniKBackend, DummyBackend
from linpyk.graph import ArmoniKGraph, DTArray, ops

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--endpoint", type=str, help="ArmoniK control plane endpoint.")
parser.add_argument("-n", "--n-dim", type=int, help="Matrix size.")
parser.add_argument("-b", "--block-size", type=int, help="Block size.")
#parser.add_argument("-pi", "--partition-init", type=str, help="Partition for initialization.")
#parser.add_argument("-pc", "--partition-compute", type=str, help="Partition for computation.")
args = parser.parse_args()

backend = ArmoniKBackend(
    args.endpoint,
    ["default", "init"],
    TaskOptions(
        max_duration=timedelta(seconds=300),
        priority=1,
        max_retries=5,
        partition_id="default",
    ),
)
# backend = DummyBackend()

g = ArmoniKGraph()

with g:
    a = ops.remote_dtarray_generator(args.n_dim, args.block_size, "diag")
    l = cholesky(a)

with backend.new_session() as sc:
    sc.run(g)
    sc.get(a)
    sc.get(l)

numpy_a = a.to_numpy()
numpy_l = l.to_numpy(triangular=True)

print(
    f"Max coefficient difference: {np.max((np.matmul(numpy_l, numpy_l.T) - numpy_a).flat)}"
)
