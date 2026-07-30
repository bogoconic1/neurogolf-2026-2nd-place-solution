from dsl import *
from constants import *


def solve_b548a754(I):
    x1 = objects(I, T, F, T)
    x2 = replace(I, EIGHT, ZERO)
    x3 = leastcolor(x2)
    x4 = replace(x2, x3, ZERO)
    x5 = leastcolor(x4)
    x6 = merge(x1)
    x7 = backdrop(x6)
    x8 = box(x6)
    x9 = fill(I, x3, x7)
    O = fill(x9, x5, x8)
    return O
