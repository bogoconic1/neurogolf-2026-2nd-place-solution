from dsl import *
from constants import *


def solve_36d67576(I):
    x1 = objects(I, F, F, T)
    x2 = argmax(x1, numcolors)
    x3 = astuple(TWO, FOUR)
    x4 = rbind(contained, x3)
    x5 = compose(x4, first)
    x6 = rbind(sfilter, x5)
    x7 = lbind(rbind, subtract)
    x8 = lbind(occurrences, I)
    x9 = lbind(lbind, shift)
    x10 = compose(x7, ulcorner)
    x11 = chain(x10, x6, normalize)
    x12 = chain(x8, x6, normalize)
    x13 = fork(apply, x11, x12)
    x14 = compose(x9, normalize)
    x15 = fork(mapply, x14, x13)
    x16 = astuple(cmirror, dmirror)
    x17 = astuple(hmirror, vmirror)
    x18 = combine(x16, x17)
    x19 = product(x18, x18)
    x20 = fork(compose, first, last)
    x21 = apply(x20, x19)
    x22 = totuple(x21)
    x23 = combine(x18, x22)
    x24 = rapply(x23, x2)
    x25 = mapply(x15, x24)
    O = paint(I, x25)
    return O
