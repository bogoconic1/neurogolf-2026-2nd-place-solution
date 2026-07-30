from dsl import *
from constants import *


def solve_780d0b14(I):
    x1 = asindices(I)
    x2 = objects(I, T, T, T)
    x3 = rbind(greater, TWO)
    x4 = compose(x3, size)
    x5 = sfilter(x2, x4)
    x6 = totuple(x5)
    x7 = apply(color, x6)
    x8 = apply(center, x6)
    x9 = pair(x7, x8)
    x10 = fill(I, ZERO, x1)
    x11 = paint(x10, x9)
    x12 = rbind(greater, ONE)
    x13 = compose(dedupe, totuple)
    x14 = chain(x12, size, x13)
    x15 = sfilter(x11, x14)
    x16 = rot90(x15)
    x17 = sfilter(x16, x14)
    O = rot270(x17)
    return O
