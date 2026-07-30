from dsl import *
from constants import *


def solve_a8d7556c(I):
    x1 = initset(ORIGIN)
    x2 = recolor(ZERO, x1)
    x3 = upscale(x2, TWO)
    x4 = occurrences(I, x3)
    x5 = lbind(shift, x3)
    x6 = mapply(x5, x4)
    x7 = fill(I, TWO, x6)
    x8 = add(SIX, SIX)
    x9 = astuple(EIGHT, x8)
    x10 = index(x7, x9)
    x11 = equality(x10, TWO)
    x12 = initset(x9)
    x13 = add(x9, DOWN)
    x14 = insert(x13, x12)
    x15 = toobject(x14, x7)
    x16 = toobject(x14, I)
    x17 = branch(x11, x16, x15)
    O = paint(x7, x17)
    return O
