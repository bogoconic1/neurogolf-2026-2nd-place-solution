from dsl import *
from constants import *


def solve_0962bcdd(I):
    x1 = leastcolor(I)
    x2 = replace(I, ZERO, x1)
    x3 = leastcolor(x2)
    x4 = ofcolor(I, x3)
    x5 = mapply(dneighbors, x4)
    x6 = fill(I, x3, x5)
    x7 = objects(x6, F, T, T)
    x8 = fork(connect, ulcorner, lrcorner)
    x9 = fork(connect, llcorner, urcorner)
    x10 = fork(combine, x8, x9)
    x11 = mapply(x10, x7)
    O = fill(x6, x1, x11)
    return O
