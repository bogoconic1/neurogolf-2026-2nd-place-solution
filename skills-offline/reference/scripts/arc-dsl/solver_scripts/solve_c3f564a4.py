from dsl import *
from constants import *


def solve_c3f564a4(I):
    x1 = asindices(I)
    x2 = dmirror(I)
    x3 = invert(NINE)
    x4 = papply(pair, I, x2)
    x5 = lbind(apply, maximum)
    x6 = apply(x5, x4)
    x7 = ofcolor(x6, ZERO)
    x8 = difference(x1, x7)
    x9 = toobject(x8, x6)
    x10 = interval(x3, NINE, ONE)
    x11 = interval(NINE, x3, NEG_ONE)
    x12 = pair(x10, x11)
    x13 = lbind(shift, x9)
    x14 = mapply(x13, x12)
    O = paint(x6, x14)
    return O
