from dsl import *
from constants import *


def solve_a2fd1cf0(I):
    x1 = ofcolor(I, TWO)
    x2 = ofcolor(I, THREE)
    x3 = uppermost(x1)
    x4 = leftmost(x1)
    x5 = uppermost(x2)
    x6 = leftmost(x2)
    x7 = astuple(x3, x5)
    x8 = minimum(x7)
    x9 = maximum(x7)
    x10 = astuple(x8, x6)
    x11 = astuple(x9, x6)
    x12 = connect(x10, x11)
    x13 = astuple(x4, x6)
    x14 = minimum(x13)
    x15 = maximum(x13)
    x16 = astuple(x3, x14)
    x17 = astuple(x3, x15)
    x18 = connect(x16, x17)
    x19 = combine(x12, x18)
    O = underfill(I, EIGHT, x19)
    return O
