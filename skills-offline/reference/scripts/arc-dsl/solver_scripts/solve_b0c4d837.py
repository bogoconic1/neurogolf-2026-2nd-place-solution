from dsl import *
from constants import *


def solve_b0c4d837(I):
    x1 = ofcolor(I, FIVE)
    x2 = ofcolor(I, EIGHT)
    x3 = height(x1)
    x4 = decrement(x3)
    x5 = height(x2)
    x6 = subtract(x4, x5)
    x7 = astuple(ONE, x6)
    x8 = canvas(EIGHT, x7)
    x9 = subtract(SIX, x6)
    x10 = astuple(ONE, x9)
    x11 = canvas(ZERO, x10)
    x12 = hconcat(x8, x11)
    x13 = hsplit(x12, TWO)
    x14 = first(x13)
    x15 = last(x13)
    x16 = vmirror(x15)
    x17 = vconcat(x14, x16)
    x18 = astuple(ONE, THREE)
    x19 = canvas(ZERO, x18)
    O = vconcat(x17, x19)
    return O
