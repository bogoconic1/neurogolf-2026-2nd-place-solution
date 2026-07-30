from dsl import *
from constants import *


def solve_2bee17df(I):
    x1 = height(I)
    x2 = rot90(I)
    x3 = subtract(x1, TWO)
    x4 = interval(ZERO, x1, ONE)
    x5 = rbind(colorcount, ZERO)
    x6 = matcher(x5, x3)
    x7 = rbind(vsplit, x1)
    x8 = lbind(apply, x6)
    x9 = compose(x8, x7)
    x10 = x9(I)
    x11 = pair(x4, x10)
    x12 = sfilter(x11, last)
    x13 = mapply(hfrontier, x12)
    x14 = x9(x2)
    x15 = pair(x14, x4)
    x16 = sfilter(x15, first)
    x17 = mapply(vfrontier, x16)
    x18 = astuple(x13, x17)
    x19 = merge(x18)
    O = underfill(I, THREE, x19)
    return O
