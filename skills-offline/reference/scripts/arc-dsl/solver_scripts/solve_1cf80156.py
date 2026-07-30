from dsl import *
from constants import *


def solve_1cf80156(I):
    x1 = objects(I, T, T, T)
    x2 = first(x1)
    O = subgrid(x2, I)
    return O
