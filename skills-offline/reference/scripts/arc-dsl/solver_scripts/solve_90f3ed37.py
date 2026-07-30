from dsl import *
from constants import *


def solve_90f3ed37(I):
    x1 = objects(I, T, T, T)
    x2 = order(x1, uppermost)
    x3 = first(x2)
    x4 = remove(x3, x2)
    x5 = normalize(x3)
    x6 = lbind(shift, x5)
    x7 = compose(x6, ulcorner)
    x8 = interval(TWO, NEG_ONE, NEG_ONE)
    x9 = apply(tojvec, x8)
    x10 = rbind(apply, x9)
    x11 = lbind(compose, size)
    x12 = lbind(lbind, intersection)
    x13 = compose(x11, x12)
    x14 = lbind(lbind, shift)
    x15 = chain(x10, x14, x7)
    x16 = fork(argmax, x15, x13)
    x17 = mapply(x16, x4)
    O = underfill(I, ONE, x17)
    return O
