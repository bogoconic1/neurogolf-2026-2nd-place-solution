from dsl import *
from constants import *


def solve_caa06a1f(I):
    x1 = asobject(I)
    x2 = shape(I)
    x3 = decrement(x2)
    x4 = index(I, x3)
    x5 = double(x2)
    x6 = canvas(x4, x5)
    x7 = paint(x6, x1)
    x8 = objects(x7, F, F, T)
    x9 = first(x8)
    x10 = shift(x9, LEFT)
    x11 = vperiod(x10)
    x12 = hperiod(x10)
    x13 = neighbors(ORIGIN)
    x14 = lbind(mapply, neighbors)
    x15 = power(x14, TWO)
    x16 = x15(x13)
    x17 = astuple(x11, x12)
    x18 = lbind(multiply, x17)
    x19 = apply(x18, x16)
    x20 = lbind(shift, x10)
    x21 = mapply(x20, x19)
    O = paint(I, x21)
    return O
