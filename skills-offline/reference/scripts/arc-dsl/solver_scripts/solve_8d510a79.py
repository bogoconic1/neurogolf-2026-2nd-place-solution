from dsl import *
from constants import *


def solve_8d510a79(I):
    x1 = ofcolor(I, ONE)
    x2 = ofcolor(I, TWO)
    x3 = ofcolor(I, FIVE)
    x4 = uppermost(x3)
    x5 = chain(toivec, decrement, double)
    x6 = lbind(greater, x4)
    x7 = compose(x6, first)
    x8 = chain(invert, x5, x7)
    x9 = fork(shoot, identity, x8)
    x10 = compose(x5, x7)
    x11 = fork(shoot, identity, x10)
    x12 = lbind(matcher, x7)
    x13 = compose(x12, x7)
    x14 = fork(sfilter, x11, x13)
    x15 = mapply(x9, x1)
    x16 = mapply(x14, x2)
    x17 = underfill(I, TWO, x16)
    O = fill(x17, ONE, x15)
    return O
