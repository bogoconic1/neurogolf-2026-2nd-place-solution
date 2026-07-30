from dsl import *
from constants import *


def solve_beb8660c(I):
    x1 = shape(I)
    x2 = objects(I, T, F, T)
    x3 = compose(invert, size)
    x4 = order(x2, x3)
    x5 = apply(normalize, x4)
    x6 = size(x5)
    x7 = interval(ZERO, x6, ONE)
    x8 = apply(toivec, x7)
    x9 = mpapply(shift, x5, x8)
    x10 = canvas(ZERO, x1)
    x11 = paint(x10, x9)
    O = rot180(x11)
    return O
