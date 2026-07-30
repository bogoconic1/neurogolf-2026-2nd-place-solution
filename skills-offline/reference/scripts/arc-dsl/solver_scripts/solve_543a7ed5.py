from dsl import *
from constants import *


def solve_543a7ed5(I):
    x1 = objects(I, T, F, T)
    x2 = colorfilter(x1, SIX)
    x3 = mapply(outbox, x2)
    x4 = fill(I, THREE, x3)
    x5 = mapply(delta, x2)
    O = fill(x4, FOUR, x5)
    return O
