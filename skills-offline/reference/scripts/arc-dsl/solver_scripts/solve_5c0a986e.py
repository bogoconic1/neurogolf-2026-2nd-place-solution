from dsl import *
from constants import *


def solve_5c0a986e(I):
    x1 = ofcolor(I, TWO)
    x2 = ofcolor(I, ONE)
    x3 = lrcorner(x1)
    x4 = ulcorner(x2)
    x5 = shoot(x3, UNITY)
    x6 = shoot(x4, NEG_UNITY)
    x7 = fill(I, TWO, x5)
    O = fill(x7, ONE, x6)
    return O
