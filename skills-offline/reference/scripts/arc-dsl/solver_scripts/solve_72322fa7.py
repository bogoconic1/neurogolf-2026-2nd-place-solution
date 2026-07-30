from dsl import *
from constants import *


def solve_72322fa7(I):
    x1 = objects(I, F, T, T)
    x2 = matcher(numcolors, ONE)
    x3 = sfilter(x1, x2)
    x4 = difference(x1, x3)
    x5 = lbind(matcher, first)
    x6 = compose(x5, mostcolor)
    x7 = fork(sfilter, identity, x6)
    x8 = fork(difference, identity, x7)
    x9 = lbind(occurrences, I)
    x10 = compose(x9, x7)
    x11 = compose(x9, x8)
    x12 = compose(ulcorner, x8)
    x13 = fork(subtract, ulcorner, x12)
    x14 = lbind(rbind, add)
    x15 = compose(x14, x13)
    x16 = fork(apply, x15, x11)
    x17 = lbind(lbind, shift)
    x18 = compose(x17, normalize)
    x19 = fork(mapply, x18, x10)
    x20 = fork(mapply, x18, x16)
    x21 = mapply(x19, x4)
    x22 = mapply(x20, x4)
    x23 = paint(I, x21)
    O = paint(x23, x22)
    return O
