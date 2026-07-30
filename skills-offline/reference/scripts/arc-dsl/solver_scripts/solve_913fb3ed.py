from dsl import *
from constants import *


def solve_913fb3ed(I):
    x1 = ofcolor(I, THREE)
    x2 = ofcolor(I, EIGHT)
    x3 = ofcolor(I, TWO)
    x4 = mapply(neighbors, x1)
    x5 = mapply(neighbors, x2)
    x6 = mapply(neighbors, x3)
    x7 = fill(I, SIX, x4)
    x8 = fill(x7, FOUR, x5)
    O = fill(x8, ONE, x6)
    return O
