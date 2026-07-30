from dsl import *
from constants import *


def solve_0b148d64(I):
    x1 = partition(I)
    x2 = argmin(x1, size)
    O = subgrid(x2, I)
    return O
