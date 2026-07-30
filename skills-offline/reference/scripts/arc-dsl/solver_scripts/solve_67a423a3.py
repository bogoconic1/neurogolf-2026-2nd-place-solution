from dsl import *
from constants import *


def solve_67a423a3(I):
    x1 = leastcolor(I)
    x2 = objects(I, T, F, T)
    x3 = colorfilter(x2, x1)
    x4 = merge(x3)
    x5 = delta(x4)
    x6 = first(x5)
    x7 = neighbors(x6)
    O = fill(I, FOUR, x7)
    return O
