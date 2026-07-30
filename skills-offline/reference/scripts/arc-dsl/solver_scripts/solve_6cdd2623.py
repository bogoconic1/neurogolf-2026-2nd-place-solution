from dsl import *
from constants import *


def solve_6cdd2623(I):
    x1 = leastcolor(I)
    x2 = ofcolor(I, x1)
    x3 = prapply(connect, x2, x2)
    x4 = fgpartition(I)
    x5 = merge(x4)
    x6 = cover(I, x5)
    x7 = fork(either, hline, vline)
    x8 = box(x5)
    x9 = rbind(difference, x8)
    x10 = chain(positive, size, x9)
    x11 = fork(both, x7, x10)
    x12 = mfilter(x3, x11)
    O = fill(x6, x1, x12)
    return O
