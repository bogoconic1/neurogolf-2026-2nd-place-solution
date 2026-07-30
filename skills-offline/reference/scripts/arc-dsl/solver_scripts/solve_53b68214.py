from dsl import *
from constants import *


def solve_53b68214(I):
    x1 = width(I)
    x2 = objects(I, T, T, T)
    x3 = first(x2)
    x4 = vperiod(x3)
    x5 = toivec(x4)
    x6 = interval(ZERO, NINE, ONE)
    x7 = lbind(multiply, x5)
    x8 = apply(x7, x6)
    x9 = lbind(shift, x3)
    x10 = mapply(x9, x8)
    x11 = astuple(x1, x1)
    x12 = portrait(x3)
    x13 = shape(x3)
    x14 = add(DOWN, x13)
    x15 = decrement(x14)
    x16 = shift(x3, x15)
    x17 = branch(x12, x10, x16)
    x18 = canvas(ZERO, x11)
    x19 = paint(x18, x3)
    O = paint(x19, x17)
    return O
