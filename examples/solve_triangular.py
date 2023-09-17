import numpy as np

from linpyk.algorithms import solve
from linpyk.backend import DummyBackend
from linpyk.graph import ArmoniKGraph, DTArray
from linpyk.utils import make_spd_system


backend = DummyBackend()

g = ArmoniKGraph()

size = 10
block_size = 2

a, x, b = make_spd_system(size)

with g:
    A = DTArray.from_numpy("A", a, block_size)
    B = DTArray.from_numpy("b", b, block_size)
    y = solve(A, B, assume_a='spd')

g.savefig("./graph.png")

with backend.new_session() as sc:
    sc.run(g)
    sc.get(y)

print(
    f"Max coefficient difference: {np.max(y.to_numpy() - x)}"
)
