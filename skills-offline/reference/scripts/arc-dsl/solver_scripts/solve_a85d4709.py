from dsl import *
from constants import *


def solve_a85d4709(I):
    x1 = ofcolor(I, FIVE)
    x2 = lbind(matcher, last)
    x3 = lbind(sfilter, x1)
    x4 = lbind(mapply, hfrontier)
    x5 = chain(x4, x3, x2)
    x6 = x5(ZERO)
    x7 = x5(TWO)
    x8 = x5(ONE)
    x9 = fill(I, TWO, x6)
    x10 = fill(x9, THREE, x7)
    O = fill(x10, FOUR, x8)
    return O
