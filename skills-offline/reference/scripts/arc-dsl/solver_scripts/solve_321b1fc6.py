from dsl import *
from constants import *


def solve_321b1fc6(I):
    x1 = objects(I, F, F, T)
    x2 = colorfilter(x1, EIGHT)
    x3 = difference(x1, x2)
    x4 = first(x3)
    x5 = cover(I, x4)
    x6 = normalize(x4)
    x7 = lbind(shift, x6)
    x8 = apply(ulcorner, x2)
    x9 = mapply(x7, x8)
    O = paint(x5, x9)
    return O
