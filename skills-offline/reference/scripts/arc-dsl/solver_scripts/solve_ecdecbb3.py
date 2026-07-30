from dsl import *
from constants import *


def solve_ecdecbb3(I):
    x1 = objects(I, T, F, T)
    x2 = colorfilter(x1, TWO)
    x3 = colorfilter(x1, EIGHT)
    x4 = product(x2, x3)
    x5 = fork(gravitate, first, last)
    x6 = compose(crement, x5)
    x7 = compose(center, first)
    x8 = fork(add, x7, x6)
    x9 = fork(connect, x7, x8)
    x10 = apply(x9, x4)
    x11 = lbind(greater, EIGHT)
    x12 = compose(x11, size)
    x13 = mfilter(x10, x12)
    x14 = fill(I, TWO, x13)
    x15 = apply(x8, x4)
    x16 = intersection(x13, x15)
    x17 = mapply(neighbors, x16)
    O = fill(x14, EIGHT, x17)
    return O
