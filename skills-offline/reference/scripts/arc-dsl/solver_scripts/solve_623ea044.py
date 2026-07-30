from dsl import *
from constants import *


def solve_623ea044(I):
    x1 = objects(I, T, F, T)
    x2 = first(x1)
    x3 = center(x2)
    x4 = color(x2)
    x5 = astuple(UNITY, NEG_UNITY)
    x6 = astuple(UP_RIGHT, DOWN_LEFT)
    x7 = combine(x5, x6)
    x8 = lbind(shoot, x3)
    x9 = mapply(x8, x7)
    O = fill(I, x4, x9)
    return O
