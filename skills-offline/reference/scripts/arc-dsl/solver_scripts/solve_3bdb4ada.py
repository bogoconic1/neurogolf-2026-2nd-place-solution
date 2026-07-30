from dsl import *
from constants import *


def solve_3bdb4ada(I):
    x1 = objects(I, T, F, T)
    x2 = totuple(x1)
    x3 = compose(increment, ulcorner)
    x4 = compose(decrement, lrcorner)
    x5 = apply(x3, x2)
    x6 = apply(x4, x2)
    x7 = papply(connect, x5, x6)
    x8 = apply(last, x5)
    x9 = compose(last, first)
    x10 = power(last, TWO)
    x11 = fork(subtract, x9, x10)
    x12 = compose(even, x11)
    x13 = lbind(rbind, astuple)
    x14 = lbind(compose, x12)
    x15 = compose(x14, x13)
    x16 = fork(sfilter, first, x15)
    x17 = pair(x7, x8)
    x18 = mapply(x16, x17)
    O = fill(I, ZERO, x18)
    return O
