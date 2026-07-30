from dsl import *
from constants import *


def solve_d0f5fe59(I):
    x1 = objects(I, T, F, T)
    x2 = size(x1)
    x3 = astuple(x2, x2)
    x4 = canvas(ZERO, x3)
    x5 = shoot(ORIGIN, UNITY)
    O = fill(x4, EIGHT, x5)
    return O
