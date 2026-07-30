from dsl import *
from constants import *


def solve_3428a4f5(I):
    x1 = tophalf(I)
    x2 = bottomhalf(I)
    x3 = astuple(SIX, FIVE)
    x4 = ofcolor(x1, TWO)
    x5 = ofcolor(x2, TWO)
    x6 = combine(x4, x5)
    x7 = intersection(x4, x5)
    x8 = difference(x6, x7)
    x9 = canvas(ZERO, x3)
    O = fill(x9, THREE, x8)
    return O
