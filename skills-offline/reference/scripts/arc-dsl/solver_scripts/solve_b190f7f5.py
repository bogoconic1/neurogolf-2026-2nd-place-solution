from dsl import *
from constants import *


def solve_b190f7f5(I):
    x1 = portrait(I)
    x2 = branch(x1, vsplit, hsplit)
    x3 = x2(I, TWO)
    x4 = argmin(x3, numcolors)
    x5 = argmax(x3, numcolors)
    x6 = width(x5)
    x7 = rbind(repeat, x6)
    x8 = chain(dmirror, merge, x7)
    x9 = upscale(x5, x6)
    x10 = x8(x4)
    x11 = x8(x10)
    x12 = ofcolor(x11, ZERO)
    O = fill(x9, ZERO, x12)
    return O
