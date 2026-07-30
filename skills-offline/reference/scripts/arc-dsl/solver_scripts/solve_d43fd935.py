from dsl import *
from constants import *


def solve_d43fd935(I):
    x1 = objects(I, T, F, T)
    x2 = ofcolor(I, THREE)
    x3 = sizefilter(x1, ONE)
    x4 = rbind(vmatching, x2)
    x5 = rbind(hmatching, x2)
    x6 = fork(either, x4, x5)
    x7 = sfilter(x3, x6)
    x8 = rbind(gravitate, x2)
    x9 = fork(add, center, x8)
    x10 = fork(connect, center, x9)
    x11 = fork(recolor, color, x10)
    x12 = mapply(x11, x7)
    O = paint(I, x12)
    return O
