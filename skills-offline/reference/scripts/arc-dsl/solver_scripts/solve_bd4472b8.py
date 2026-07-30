from dsl import *
from constants import *


def solve_bd4472b8(I):
    x1 = width(I)
    x2 = astuple(TWO, x1)
    x3 = crop(I, ORIGIN, x2)
    x4 = tophalf(x3)
    x5 = dmirror(x4)
    x6 = hupscale(x5, x1)
    x7 = repeat(x6, TWO)
    x8 = merge(x7)
    O = vconcat(x3, x8)
    return O
