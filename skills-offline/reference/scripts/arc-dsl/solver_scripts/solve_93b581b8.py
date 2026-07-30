from dsl import *
from constants import *


def solve_93b581b8(I):
    x1 = fgpartition(I)
    x2 = chain(cmirror, dmirror, merge)
    x3 = x2(x1)
    x4 = upscale(x3, THREE)
    x5 = astuple(NEG_TWO, NEG_TWO)
    x6 = shift(x4, x5)
    x7 = underpaint(I, x6)
    x8 = toindices(x3)
    x9 = fork(combine, hfrontier, vfrontier)
    x10 = mapply(x9, x8)
    x11 = difference(x10, x8)
    O = fill(x7, ZERO, x11)
    return O
