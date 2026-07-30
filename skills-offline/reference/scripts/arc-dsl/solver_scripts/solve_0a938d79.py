from dsl import *
from constants import *


def solve_0a938d79(I):
    x1 = portrait(I)
    x2 = branch(x1, dmirror, identity)
    x3 = x2(I)
    x4 = fgpartition(x3)
    x5 = merge(x4)
    x6 = chain(double, decrement, width)
    x7 = x6(x5)
    x8 = compose(vfrontier, tojvec)
    x9 = lbind(mapply, x8)
    x10 = rbind(interval, x7)
    x11 = width(x3)
    x12 = rbind(x10, x11)
    x13 = chain(x9, x12, leftmost)
    x14 = fork(recolor, color, x13)
    x15 = mapply(x14, x4)
    x16 = paint(x3, x15)
    O = x2(x16)
    return O
