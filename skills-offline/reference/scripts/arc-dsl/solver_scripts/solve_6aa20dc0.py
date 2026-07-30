from dsl import *
from constants import *


def solve_6aa20dc0(I):
    x1 = objects(I, F, T, T)
    x2 = argmax(x1, numcolors)
    x3 = normalize(x2)
    x4 = lbind(matcher, first)
    x5 = compose(x4, mostcolor)
    x6 = fork(sfilter, identity, x5)
    x7 = fork(difference, identity, x6)
    x8 = lbind(rbind, upscale)
    x9 = interval(ONE, FOUR, ONE)
    x10 = apply(x8, x9)
    x11 = initset(identity)
    x12 = insert(vmirror, x11)
    x13 = insert(hmirror, x12)
    x14 = insert(cmirror, x13)
    x15 = insert(dmirror, x14)
    x16 = fork(compose, first, last)
    x17 = lbind(occurrences, I)
    x18 = lbind(lbind, shift)
    x19 = compose(x17, x7)
    x20 = product(x15, x10)
    x21 = apply(x16, x20)
    x22 = rapply(x21, x3)
    x23 = fork(mapply, x18, x19)
    x24 = mapply(x23, x22)
    O = paint(I, x24)
    return O
