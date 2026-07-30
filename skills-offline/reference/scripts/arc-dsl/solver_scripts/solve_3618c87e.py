from dsl import *
from constants import *


def solve_3618c87e(I):
    x1 = objects(I, T, F, T)
    x2 = sizefilter(x1, ONE)
    x3 = merge(x2)
    O = move(I, x3, TWO_BY_ZERO)
    return O
