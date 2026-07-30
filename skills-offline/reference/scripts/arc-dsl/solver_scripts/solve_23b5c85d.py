from dsl import *
from constants import *


def solve_23b5c85d(I):
    x1 = objects(I, T, T, T)
    x2 = argmin(x1, size)
    O = subgrid(x2, I)
    return O
