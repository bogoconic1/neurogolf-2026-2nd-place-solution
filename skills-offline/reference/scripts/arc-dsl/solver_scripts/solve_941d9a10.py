from dsl import *
from constants import *


def solve_941d9a10(I):
    x1 = shape(I)
    x2 = objects(I, T, F, F)
    x3 = colorfilter(x2, ZERO)
    x4 = apply(toindices, x3)
    x5 = lbind(lbind, contained)
    x6 = lbind(extract, x4)
    x7 = compose(x6, x5)
    x8 = decrement(x1)
    x9 = astuple(FIVE, FIVE)
    x10 = x7(ORIGIN)
    x11 = x7(x8)
    x12 = x7(x9)
    x13 = fill(I, ONE, x10)
    x14 = fill(x13, THREE, x11)
    O = fill(x14, TWO, x12)
    return O
