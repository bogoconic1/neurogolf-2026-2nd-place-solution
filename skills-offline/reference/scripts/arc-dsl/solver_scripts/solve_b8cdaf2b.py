from dsl import *
from constants import *


def solve_b8cdaf2b(I):
    x1 = leastcolor(I)
    x2 = ofcolor(I, x1)
    x3 = shift(x2, UP)
    x4 = ulcorner(x3)
    x5 = urcorner(x3)
    x6 = shoot(x4, NEG_UNITY)
    x7 = shoot(x5, UP_RIGHT)
    x8 = combine(x6, x7)
    O = underfill(I, x1, x8)
    return O
