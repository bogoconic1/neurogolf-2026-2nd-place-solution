from dsl import *
from constants import *


def solve_3345333e(I):
    x1 = leastcolor(I)
    x2 = ofcolor(I, x1)
    x3 = cover(I, x2)
    x4 = leastcolor(x3)
    x5 = ofcolor(x3, x4)
    x6 = neighbors(ORIGIN)
    x7 = mapply(neighbors, x6)
    x8 = vmirror(x5)
    x9 = lbind(shift, x8)
    x10 = apply(x9, x7)
    x11 = rbind(intersection, x5)
    x12 = compose(size, x11)
    x13 = argmax(x10, x12)
    O = fill(x3, x4, x13)
    return O
