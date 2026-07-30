from dsl import *
from constants import *


def solve_1a07d186(I):
    x1 = objects(I, T, F, T)
    x2 = sizefilter(x1, ONE)
    x3 = difference(x1, x2)
    x4 = apply(color, x3)
    x5 = rbind(contained, x4)
    x6 = compose(x5, color)
    x7 = sfilter(x2, x6)
    x8 = lbind(colorfilter, x3)
    x9 = chain(first, x8, color)
    x10 = fork(gravitate, identity, x9)
    x11 = fork(shift, identity, x10)
    x12 = mapply(x11, x7)
    x13 = merge(x2)
    x14 = cover(I, x13)
    O = paint(x14, x12)
    return O
