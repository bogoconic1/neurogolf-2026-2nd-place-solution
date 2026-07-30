from dsl import *
from constants import *


def solve_3befdf3e(I):
    x1 = objects(I, F, F, T)
    x2 = leastcolor(I)
    x3 = palette(I)
    x4 = remove(ZERO, x3)
    x5 = other(x4, x2)
    x6 = switch(I, x2, x5)
    x7 = compose(width, inbox)
    x8 = lbind(power, outbox)
    x9 = compose(x8, x7)
    x10 = initset(x9)
    x11 = lbind(rapply, x10)
    x12 = chain(initset, first, x11)
    x13 = fork(rapply, x12, identity)
    x14 = compose(first, x13)
    x15 = compose(backdrop, x14)
    x16 = lbind(chain, backdrop)
    x17 = lbind(x16, inbox)
    x18 = compose(x17, x9)
    x19 = lbind(apply, initset)
    x20 = chain(x19, corners, x15)
    x21 = fork(mapply, x18, x20)
    x22 = fork(intersection, x15, x21)
    x23 = mapply(x15, x1)
    x24 = mapply(x22, x1)
    x25 = underfill(x6, x5, x23)
    O = fill(x25, ZERO, x24)
    return O
