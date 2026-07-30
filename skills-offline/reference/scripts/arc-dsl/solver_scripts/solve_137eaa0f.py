from dsl import *
from constants import *


def solve_137eaa0f(I):
    x1 = objects(I, F, T, T)
    x2 = matcher(first, FIVE)
    x3 = rbind(sfilter, x2)
    x4 = chain(invert, center, x3)
    x5 = fork(shift, identity, x4)
    x6 = canvas(ZERO, THREE_BY_THREE)
    x7 = mapply(x5, x1)
    x8 = shift(x7, UNITY)
    O = paint(x6, x8)
    return O
