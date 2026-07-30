from dsl import *
from constants import *


def solve_28bf18c6(I):
    x1 = objects(I, T, T, T)
    x2 = first(x1)
    x3 = subgrid(x2, I)
    O = hconcat(x3, x3)
    return O
