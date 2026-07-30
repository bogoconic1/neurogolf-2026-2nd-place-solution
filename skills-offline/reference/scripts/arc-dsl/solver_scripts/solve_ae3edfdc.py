from dsl import *
from constants import *


def solve_ae3edfdc(I):
    x1 = objects(I, T, F, T)
    x2 = replace(I, THREE, ZERO)
    x3 = replace(x2, SEVEN, ZERO)
    x4 = lbind(colorfilter, x1)
    x5 = lbind(rbind, gravitate)
    x6 = chain(x5, first, x4)
    x7 = x6(TWO)
    x8 = x6(ONE)
    x9 = x4(THREE)
    x10 = x4(SEVEN)
    x11 = fork(shift, identity, x7)
    x12 = fork(shift, identity, x8)
    x13 = mapply(x11, x9)
    x14 = mapply(x12, x10)
    x15 = paint(x3, x13)
    O = paint(x15, x14)
    return O
