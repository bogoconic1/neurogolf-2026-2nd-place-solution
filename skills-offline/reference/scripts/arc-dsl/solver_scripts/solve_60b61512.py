from dsl import *
from constants import *


def solve_60b61512(I):
    x1 = objects(I, T, T, T)
    x2 = mapply(delta, x1)
    O = fill(I, SEVEN, x2)
    return O
