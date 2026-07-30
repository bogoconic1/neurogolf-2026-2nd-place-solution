from dsl import *
from constants import *


def solve_a48eeaf7(I):
    x1 = ofcolor(I, TWO)
    x2 = outbox(x1)
    x3 = apply(initset, x2)
    x4 = ofcolor(I, FIVE)
    x5 = lbind(argmin, x3)
    x6 = lbind(lbind, manhattan)
    x7 = compose(x6, initset)
    x8 = compose(x5, x7)
    x9 = mapply(x8, x4)
    x10 = cover(I, x4)
    O = fill(x10, FIVE, x9)
    return O
