from dsl import *
from constants import *


def solve_88a10436(I):
    x1 = objects(I, F, F, T)
    x2 = colorfilter(x1, FIVE)
    x3 = first(x2)
    x4 = center(x3)
    x5 = difference(x1, x2)
    x6 = first(x5)
    x7 = normalize(x6)
    x8 = shift(x7, x4)
    x9 = shift(x8, NEG_UNITY)
    O = paint(I, x9)
    return O
