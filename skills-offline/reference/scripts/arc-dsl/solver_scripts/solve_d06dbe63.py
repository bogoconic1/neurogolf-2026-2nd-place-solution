from dsl import *
from constants import *


def solve_d06dbe63(I):
    x1 = ofcolor(I, EIGHT)
    x2 = center(x1)
    x3 = connect(ORIGIN, DOWN)
    x4 = connect(ORIGIN, ZERO_BY_TWO)
    x5 = combine(x3, x4)
    x6 = subtract(x2, TWO_BY_ZERO)
    x7 = shift(x5, x6)
    x8 = astuple(NEG_TWO, TWO)
    x9 = interval(ZERO, FIVE, ONE)
    x10 = lbind(multiply, x8)
    x11 = apply(x10, x9)
    x12 = lbind(shift, x7)
    x13 = mapply(x12, x11)
    x14 = fill(I, FIVE, x13)
    x15 = rot180(x14)
    x16 = ofcolor(x15, EIGHT)
    x17 = center(x16)
    x18 = subtract(x17, x6)
    x19 = shift(x13, x18)
    x20 = toivec(NEG_TWO)
    x21 = shift(x19, x20)
    x22 = fill(x15, FIVE, x21)
    O = rot180(x22)
    return O
