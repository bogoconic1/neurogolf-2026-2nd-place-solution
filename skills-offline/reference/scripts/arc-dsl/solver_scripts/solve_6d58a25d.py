from dsl import *
from constants import *


def solve_6d58a25d(I):
    x1 = objects(I, T, T, T)
    x2 = argmax(x1, size)
    x3 = remove(x2, x1)
    x4 = merge(x3)
    x5 = color(x4)
    x6 = uppermost(x2)
    x7 = rbind(greater, x6)
    x8 = compose(x7, uppermost)
    x9 = rbind(vmatching, x2)
    x10 = fork(both, x9, x8)
    x11 = sfilter(x3, x10)
    x12 = increment(x6)
    x13 = rbind(greater, x12)
    x14 = compose(x13, first)
    x15 = rbind(sfilter, x14)
    x16 = chain(x15, vfrontier, center)
    x17 = mapply(x16, x11)
    O = underfill(I, x5, x17)
    return O
