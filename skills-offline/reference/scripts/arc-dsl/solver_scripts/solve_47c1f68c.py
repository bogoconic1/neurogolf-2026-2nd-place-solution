from dsl import *
from constants import *


def solve_47c1f68c(I):
    x1 = leastcolor(I)
    x2 = vmirror(I)
    x3 = objects(I, T, T, T)
    x4 = merge(x3)
    x5 = mostcolor(x4)
    x6 = cellwise(I, x2, x1)
    x7 = hmirror(x6)
    x8 = cellwise(x6, x7, x1)
    x9 = compress(x8)
    O = replace(x9, x1, x5)
    return O
