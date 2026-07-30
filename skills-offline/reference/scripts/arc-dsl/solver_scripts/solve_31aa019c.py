from dsl import *
from constants import *


def solve_31aa019c(I):
    x1 = leastcolor(I)
    x2 = ofcolor(I, x1)
    x3 = first(x2)
    x4 = neighbors(x3)
    x5 = astuple(TEN, TEN)
    x6 = canvas(ZERO, x5)
    x7 = initset(x3)
    x8 = fill(x6, x1, x7)
    O = fill(x8, TWO, x4)
    return O
