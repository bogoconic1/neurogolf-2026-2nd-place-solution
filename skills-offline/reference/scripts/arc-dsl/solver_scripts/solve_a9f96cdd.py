from dsl import *
from constants import *


def solve_a9f96cdd(I):
    x1 = ofcolor(I, TWO)
    x2 = replace(I, TWO, ZERO)
    x3 = shift(x1, NEG_UNITY)
    x4 = fill(x2, THREE, x3)
    x5 = shift(x1, UP_RIGHT)
    x6 = fill(x4, SIX, x5)
    x7 = shift(x1, DOWN_LEFT)
    x8 = fill(x6, EIGHT, x7)
    x9 = shift(x1, UNITY)
    O = fill(x8, SEVEN, x9)
    return O
