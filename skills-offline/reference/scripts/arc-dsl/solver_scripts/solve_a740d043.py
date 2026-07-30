from dsl import *
from constants import *


def solve_a740d043(I):
    x1 = objects(I, T, T, T)
    x2 = merge(x1)
    x3 = subgrid(x2, I)
    O = replace(x3, ONE, ZERO)
    return O
