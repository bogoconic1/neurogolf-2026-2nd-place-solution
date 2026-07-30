from dsl import *
from constants import *


def solve_98cf29f8(I):
    x1 = fgpartition(I)
    x2 = fork(multiply, height, width)
    x3 = fork(equality, size, x2)
    x4 = extract(x1, x3)
    x5 = other(x1, x4)
    x6 = color(x5)
    x7 = rbind(greater, THREE)
    x8 = rbind(toobject, I)
    x9 = rbind(colorcount, x6)
    x10 = chain(x8, ineighbors, last)
    x11 = chain(x7, x9, x10)
    x12 = sfilter(x5, x11)
    x13 = outbox(x12)
    x14 = backdrop(x13)
    x15 = cover(I, x5)
    x16 = gravitate(x14, x4)
    x17 = shift(x14, x16)
    O = fill(x15, x6, x17)
    return O
