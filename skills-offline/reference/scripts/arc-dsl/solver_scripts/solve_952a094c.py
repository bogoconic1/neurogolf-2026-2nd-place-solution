from dsl import *
from constants import *


def solve_952a094c(I):
    x1 = objects(I, T, F, T)
    x2 = sizefilter(x1, ONE)
    x3 = argmax(x1, size)
    x4 = outbox(x3)
    x5 = corners(x4)
    x6 = lbind(rbind, manhattan)
    x7 = lbind(argmax, x2)
    x8 = chain(x7, x6, initset)
    x9 = compose(color, x8)
    x10 = fork(astuple, x9, identity)
    x11 = apply(x10, x5)
    x12 = merge(x2)
    x13 = cover(I, x12)
    O = paint(x13, x11)
    return O
