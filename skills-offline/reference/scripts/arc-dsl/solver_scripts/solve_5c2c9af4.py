from dsl import *
from constants import *


def solve_5c2c9af4(I):
    x1 = leastcolor(I)
    x2 = ofcolor(I, x1)
    x3 = center(x2)
    x4 = ulcorner(x2)
    x5 = subtract(x3, x4)
    x6 = multiply(NEG_ONE, NINE)
    x7 = interval(ZERO, NINE, ONE)
    x8 = interval(ZERO, x6, NEG_ONE)
    x9 = lbind(multiply, x5)
    x10 = apply(x9, x7)
    x11 = apply(x9, x8)
    x12 = pair(x10, x11)
    x13 = mapply(box, x12)
    x14 = shift(x13, x3)
    O = fill(I, x1, x14)
    return O
