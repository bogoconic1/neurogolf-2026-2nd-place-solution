from dsl import *
from constants import *


def solve_bdad9b1f(I):
    x1 = ofcolor(I, TWO)
    x2 = ofcolor(I, EIGHT)
    x3 = center(x1)
    x4 = center(x2)
    x5 = hfrontier(x3)
    x6 = vfrontier(x4)
    x7 = intersection(x5, x6)
    x8 = fill(I, TWO, x5)
    x9 = fill(x8, EIGHT, x6)
    O = fill(x9, FOUR, x7)
    return O
