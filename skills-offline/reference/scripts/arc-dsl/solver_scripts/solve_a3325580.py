from dsl import *
from constants import *


def solve_a3325580(I):
    x1 = objects(I, T, F, T)
    x2 = valmax(x1, size)
    x3 = sizefilter(x1, x2)
    x4 = order(x3, leftmost)
    x5 = apply(color, x4)
    x6 = astuple(ONE, x2)
    x7 = rbind(canvas, x6)
    x8 = apply(x7, x5)
    x9 = merge(x8)
    O = dmirror(x9)
    return O
