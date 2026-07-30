from dsl import *
from constants import *


def solve_2c608aff(I):
    x1 = leastcolor(I)
    x2 = objects(I, T, F, T)
    x3 = argmax(x2, size)
    x4 = toindices(x3)
    x5 = ofcolor(I, x1)
    x6 = prapply(connect, x4, x5)
    x7 = fork(either, vline, hline)
    x8 = mfilter(x6, x7)
    O = underfill(I, x1, x8)
    return O
