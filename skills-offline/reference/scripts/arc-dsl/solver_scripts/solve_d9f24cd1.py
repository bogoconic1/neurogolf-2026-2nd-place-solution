from dsl import *
from constants import *


def solve_d9f24cd1(I):
    x1 = ofcolor(I, TWO)
    x2 = ofcolor(I, FIVE)
    x3 = prapply(connect, x1, x2)
    x4 = mfilter(x3, vline)
    x5 = underfill(I, TWO, x4)
    x6 = matcher(numcolors, TWO)
    x7 = objects(x5, F, F, T)
    x8 = sfilter(x7, x6)
    x9 = difference(x7, x8)
    x10 = colorfilter(x9, TWO)
    x11 = mapply(toindices, x10)
    x12 = apply(urcorner, x8)
    x13 = shift(x12, UNITY)
    x14 = rbind(shoot, UP)
    x15 = mapply(x14, x13)
    x16 = fill(x5, TWO, x15)
    x17 = mapply(vfrontier, x11)
    O = fill(x16, TWO, x17)
    return O
