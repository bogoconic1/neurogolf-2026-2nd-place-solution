from dsl import *
from constants import *


def solve_447fd412(I):
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
    x11 = lbind(recolor, ZERO)
    x12 = compose(x11, outbox)
    x13 = fork(combine, identity, x12)
    x14 = lbind(occurrences, I)
    x15 = lbind(rbind, subtract)
    x16 = lbind(apply, increment)
    x17 = lbind(lbind, shift)
    x18 = chain(x15, ulcorner, x7)
    x19 = chain(x14, x13, x7)
    x20 = fork(apply, x18, x19)
    x21 = compose(x16, x20)
    x22 = fork(mapply, x17, x21)
    x23 = rapply(x10, x3)
    x24 = mapply(x22, x23)
    O = paint(I, x24)
    return O
