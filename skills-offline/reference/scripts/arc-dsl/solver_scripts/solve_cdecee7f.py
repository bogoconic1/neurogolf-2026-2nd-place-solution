from dsl import *
from constants import *


def solve_cdecee7f(I):
    x1 = objects(I, T, F, T)
    x2 = astuple(ONE, THREE)
    x3 = size(x1)
    x4 = order(x1, leftmost)
    x5 = apply(color, x4)
    x6 = rbind(canvas, UNITY)
    x7 = apply(x6, x5)
    x8 = merge(x7)
    x9 = dmirror(x8)
    x10 = subtract(NINE, x3)
    x11 = astuple(ONE, x10)
    x12 = canvas(ZERO, x11)
    x13 = hconcat(x9, x12)
    x14 = hsplit(x13, THREE)
    x15 = merge(x14)
    x16 = crop(x15, ORIGIN, x2)
    x17 = crop(x15, DOWN, x2)
    x18 = crop(x15, TWO_BY_ZERO, x2)
    x19 = vmirror(x17)
    x20 = vconcat(x16, x19)
    O = vconcat(x20, x18)
    return O
