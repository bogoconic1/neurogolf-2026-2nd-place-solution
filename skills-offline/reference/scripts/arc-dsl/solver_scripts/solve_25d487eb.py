from dsl import *
from constants import *


def solve_25d487eb(I):
    x1 = leastcolor(I)
    x2 = objects(I, T, F, T)
    x3 = ofcolor(I, x1)
    x4 = center(x3)
    x5 = merge(x2)
    x6 = center(x5)
    x7 = subtract(x6, x4)
    x8 = shoot(x4, x7)
    O = underfill(I, x1, x8)
    return O
