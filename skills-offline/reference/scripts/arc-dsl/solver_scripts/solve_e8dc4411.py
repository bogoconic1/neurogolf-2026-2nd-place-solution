from dsl import *
from constants import *


def solve_e8dc4411(I):
    x1 = leastcolor(I)
    x2 = ofcolor(I, ZERO)
    x3 = ofcolor(I, x1)
    x4 = position(x2, x3)
    x5 = fork(connect, ulcorner, lrcorner)
    x6 = x5(x2)
    x7 = intersection(x2, x6)
    x8 = equality(x6, x7)
    x9 = fork(subtract, identity, crement)
    x10 = fork(add, identity, x9)
    x11 = branch(x8, identity, x10)
    x12 = shape(x2)
    x13 = multiply(x12, x4)
    x14 = apply(x11, x13)
    x15 = interval(ONE, FIVE, ONE)
    x16 = lbind(multiply, x14)
    x17 = apply(x16, x15)
    x18 = lbind(shift, x2)
    x19 = mapply(x18, x17)
    O = fill(I, x1, x19)
    return O
