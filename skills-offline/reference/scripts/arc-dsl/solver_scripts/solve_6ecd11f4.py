from dsl import *
from constants import *


def solve_6ecd11f4(I):
    x1 = objects(I, F, T, T)
    x2 = argmax(x1, size)
    x3 = argmin(x1, size)
    x4 = subgrid(x2, I)
    x5 = subgrid(x3, I)
    x6 = width(x4)
    x7 = width(x5)
    x8 = divide(x6, x7)
    x9 = downscale(x4, x8)
    x10 = ofcolor(x9, ZERO)
    O = fill(x5, ZERO, x10)
    return O
