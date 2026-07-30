from dsl import *
from constants import *


def solve_6c434453(I):
    x1 = objects(I, T, F, T)
    x2 = sizefilter(x1, EIGHT)
    x3 = dneighbors(UNITY)
    x4 = insert(UNITY, x3)
    x5 = merge(x2)
    x6 = cover(I, x5)
    x7 = apply(ulcorner, x2)
    x8 = lbind(shift, x4)
    x9 = mapply(x8, x7)
    O = fill(x6, TWO, x9)
    return O
