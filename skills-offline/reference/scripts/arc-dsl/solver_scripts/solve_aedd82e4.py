from dsl import *
from constants import *


def solve_aedd82e4(I):
    x1 = objects(I, T, F, F)
    x2 = colorfilter(x1, TWO)
    x3 = sizefilter(x2, ONE)
    x4 = merge(x3)
    O = fill(I, ONE, x4)
    return O
