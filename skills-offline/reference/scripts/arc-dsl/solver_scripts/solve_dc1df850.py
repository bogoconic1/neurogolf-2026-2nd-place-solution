from dsl import *
from constants import *


def solve_dc1df850(I):
    x1 = objects(I, T, F, T)
    x2 = colorfilter(x1, TWO)
    x3 = mapply(outbox, x2)
    O = fill(I, ONE, x3)
    return O
