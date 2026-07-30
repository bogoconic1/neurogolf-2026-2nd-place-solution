from dsl import *
from constants import *


def solve_e5062a87(I):
    x1 = ofcolor(I, TWO)
    x2 = recolor(ZERO, x1)
    x3 = normalize(x2)
    x4 = occurrences(I, x2)
    x5 = lbind(shift, x3)
    x6 = apply(x5, x4)
    x7 = astuple(ONE, THREE)
    x8 = astuple(FIVE, ONE)
    x9 = astuple(TWO, SIX)
    x10 = initset(x7)
    x11 = insert(x8, x10)
    x12 = insert(x9, x11)
    x13 = rbind(contained, x12)
    x14 = chain(flip, x13, ulcorner)
    x15 = sfilter(x6, x14)
    x16 = merge(x15)
    x17 = recolor(TWO, x16)
    O = paint(I, x17)
    return O
