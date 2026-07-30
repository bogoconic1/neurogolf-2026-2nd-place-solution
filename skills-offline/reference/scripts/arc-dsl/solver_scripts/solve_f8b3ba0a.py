from dsl import *
from constants import *


def solve_f8b3ba0a(I):
    x1 = compress(I)
    x2 = astuple(THREE, ONE)
    x3 = palette(x1)
    x4 = lbind(colorcount, x1)
    x5 = compose(invert, x4)
    x6 = order(x3, x5)
    x7 = rbind(canvas, UNITY)
    x8 = apply(x7, x6)
    x9 = merge(x8)
    O = crop(x9, DOWN, x2)
    return O
