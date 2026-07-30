from dsl import *
from constants import *


def solve_b6afb2da(I):
    x1 = objects(I, T, F, F)
    x2 = replace(I, FIVE, TWO)
    x3 = colorfilter(x1, FIVE)
    x4 = mapply(box, x3)
    x5 = fill(x2, FOUR, x4)
    x6 = mapply(corners, x3)
    O = fill(x5, ONE, x6)
    return O
