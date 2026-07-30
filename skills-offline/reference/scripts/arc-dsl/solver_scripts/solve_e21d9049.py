from dsl import *
from constants import *


def solve_e21d9049(I):
    x1 = asindices(I)
    x2 = leastcolor(I)
    x3 = objects(I, T, F, T)
    x4 = ofcolor(I, x2)
    x5 = merge(x3)
    x6 = shape(x5)
    x7 = neighbors(ORIGIN)
    x8 = lbind(mapply, neighbors)
    x9 = power(x8, TWO)
    x10 = x9(x7)
    x11 = lbind(multiply, x6)
    x12 = lbind(shift, x5)
    x13 = apply(x11, x10)
    x14 = mapply(x12, x13)
    x15 = lbind(hmatching, x4)
    x16 = lbind(vmatching, x4)
    x17 = fork(either, x15, x16)
    x18 = compose(x17, initset)
    x19 = paint(I, x14)
    x20 = sfilter(x1, x18)
    x21 = difference(x1, x20)
    O = cover(x19, x21)
    return O
