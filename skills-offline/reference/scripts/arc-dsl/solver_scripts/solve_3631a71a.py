from dsl import *
from constants import *


def solve_3631a71a(I):
    x1 = shape(I)
    x2 = replace(I, NINE, ZERO)
    x3 = lbind(apply, maximum)
    x4 = dmirror(x2)
    x5 = papply(pair, x2, x4)
    x6 = apply(x3, x5)
    x7 = subtract(x1, TWO_BY_TWO)
    x8 = crop(x6, TWO_BY_TWO, x7)
    x9 = vmirror(x8)
    x10 = objects(x9, T, F, T)
    x11 = merge(x10)
    x12 = shift(x11, TWO_BY_TWO)
    O = paint(x6, x12)
    return O
