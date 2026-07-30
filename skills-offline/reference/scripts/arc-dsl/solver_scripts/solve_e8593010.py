from dsl import *
from constants import *


def solve_e8593010(I):
    x1 = objects(I, T, F, T)
    x2 = sizefilter(x1, ONE)
    x3 = sizefilter(x1, TWO)
    x4 = merge(x2)
    x5 = fill(I, THREE, x4)
    x6 = merge(x3)
    x7 = fill(x5, TWO, x6)
    O = replace(x7, ZERO, ONE)
    return O
