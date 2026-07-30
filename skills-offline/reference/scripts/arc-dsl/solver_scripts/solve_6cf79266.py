from dsl import *
from constants import *


def solve_6cf79266(I):
    x1 = ofcolor(I, ZERO)
    x2 = astuple(ZERO, ORIGIN)
    x3 = initset(x2)
    x4 = upscale(x3, THREE)
    x5 = toindices(x4)
    x6 = lbind(shift, x5)
    x7 = rbind(difference, x1)
    x8 = chain(size, x7, x6)
    x9 = matcher(x8, ZERO)
    x10 = lbind(add, NEG_UNITY)
    x11 = chain(flip, x9, x10)
    x12 = fork(both, x9, x11)
    x13 = sfilter(x1, x12)
    x14 = mapply(x6, x13)
    O = fill(I, ONE, x14)
    return O
