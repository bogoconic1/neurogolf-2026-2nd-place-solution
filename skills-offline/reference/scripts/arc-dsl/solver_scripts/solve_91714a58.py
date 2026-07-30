from dsl import *
from constants import *


def solve_91714a58(I):
    x1 = shape(I)
    x2 = asindices(I)
    x3 = objects(I, T, F, T)
    x4 = argmax(x3, size)
    x5 = mostcolor(x4)
    x6 = canvas(ZERO, x1)
    x7 = paint(x6, x4)
    x8 = rbind(toobject, x7)
    x9 = rbind(colorcount, x5)
    x10 = chain(x9, x8, neighbors)
    x11 = lbind(greater, THREE)
    x12 = compose(x11, x10)
    x13 = sfilter(x2, x12)
    O = fill(x7, ZERO, x13)
    return O
