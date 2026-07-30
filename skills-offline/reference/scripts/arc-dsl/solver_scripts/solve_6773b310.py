from dsl import *
from constants import *


def solve_6773b310(I):
    x1 = compress(I)
    x2 = neighbors(ORIGIN)
    x3 = insert(ORIGIN, x2)
    x4 = rbind(multiply, THREE)
    x5 = apply(x4, x3)
    x6 = astuple(FOUR, FOUR)
    x7 = shift(x5, x6)
    x8 = fork(insert, identity, neighbors)
    x9 = apply(x8, x7)
    x10 = rbind(toobject, x1)
    x11 = apply(x10, x9)
    x12 = rbind(colorcount, SIX)
    x13 = matcher(x12, TWO)
    x14 = mfilter(x11, x13)
    x15 = fill(x1, ONE, x14)
    x16 = replace(x15, SIX, ZERO)
    O = downscale(x16, THREE)
    return O
