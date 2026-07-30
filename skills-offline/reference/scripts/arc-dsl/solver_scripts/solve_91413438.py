from dsl import *
from constants import *


def solve_91413438(I):
    x1 = colorcount(I, ZERO)
    x2 = subtract(NINE, x1)
    x3 = multiply(x1, THREE)
    x4 = multiply(x3, x1)
    x5 = subtract(x4, THREE)
    x6 = astuple(THREE, x5)
    x7 = canvas(ZERO, x6)
    x8 = hconcat(I, x7)
    x9 = objects(x8, T, T, T)
    x10 = first(x9)
    x11 = lbind(shift, x10)
    x12 = compose(x11, tojvec)
    x13 = interval(ZERO, x2, ONE)
    x14 = rbind(multiply, THREE)
    x15 = apply(x14, x13)
    x16 = mapply(x12, x15)
    x17 = paint(x8, x16)
    x18 = hsplit(x17, x1)
    O = merge(x18)
    return O
