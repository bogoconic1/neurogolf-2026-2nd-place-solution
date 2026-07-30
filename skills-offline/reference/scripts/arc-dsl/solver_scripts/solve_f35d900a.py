from dsl import *
from constants import *


def solve_f35d900a(I):
    x1 = objects(I, T, F, T)
    x2 = palette(I)
    x3 = remove(ZERO, x2)
    x4 = lbind(other, x3)
    x5 = compose(x4, color)
    x6 = fork(recolor, x5, outbox)
    x7 = mapply(x6, x1)
    x8 = mapply(toindices, x1)
    x9 = box(x8)
    x10 = difference(x9, x8)
    x11 = lbind(argmin, x8)
    x12 = rbind(compose, initset)
    x13 = lbind(rbind, manhattan)
    x14 = chain(x12, x13, initset)
    x15 = chain(initset, x11, x14)
    x16 = fork(manhattan, initset, x15)
    x17 = compose(even, x16)
    x18 = sfilter(x10, x17)
    x19 = paint(I, x7)
    O = fill(x19, FIVE, x18)
    return O
