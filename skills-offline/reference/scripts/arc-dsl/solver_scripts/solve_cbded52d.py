from dsl import *
from constants import *


def solve_cbded52d(I):
    x1 = objects(I, T, F, T)
    x2 = sizefilter(x1, ONE)
    x3 = product(x2, x2)
    x4 = fork(vmatching, first, last)
    x5 = fork(hmatching, first, last)
    x6 = fork(either, x4, x5)
    x7 = sfilter(x3, x6)
    x8 = compose(center, first)
    x9 = compose(center, last)
    x10 = fork(connect, x8, x9)
    x11 = chain(initset, center, x10)
    x12 = compose(color, first)
    x13 = fork(recolor, x12, x11)
    x14 = mapply(x13, x7)
    O = paint(I, x14)
    return O
