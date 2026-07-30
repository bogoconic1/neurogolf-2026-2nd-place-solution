from dsl import *
from constants import *


def solve_1f0c79e5(I):
    x1 = ofcolor(I, TWO)
    x2 = replace(I, TWO, ZERO)
    x3 = leastcolor(x2)
    x4 = ofcolor(x2, x3)
    x5 = combine(x1, x4)
    x6 = recolor(x3, x5)
    x7 = compose(decrement, double)
    x8 = ulcorner(x5)
    x9 = invert(x8)
    x10 = shift(x1, x9)
    x11 = apply(x7, x10)
    x12 = interval(ZERO, NINE, ONE)
    x13 = prapply(multiply, x11, x12)
    x14 = lbind(shift, x6)
    x15 = mapply(x14, x13)
    O = paint(I, x15)
    return O
