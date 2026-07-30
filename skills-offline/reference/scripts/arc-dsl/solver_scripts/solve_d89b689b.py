from dsl import *
from constants import *


def solve_d89b689b(I):
    x1 = objects(I, T, F, T)
    x2 = ofcolor(I, EIGHT)
    x3 = sizefilter(x1, ONE)
    x4 = apply(initset, x2)
    x5 = lbind(argmin, x4)
    x6 = lbind(rbind, manhattan)
    x7 = compose(x5, x6)
    x8 = fork(recolor, color, x7)
    x9 = mapply(x8, x3)
    x10 = merge(x3)
    x11 = cover(I, x10)
    O = paint(x11, x9)
    return O
