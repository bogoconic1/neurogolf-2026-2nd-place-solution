from dsl import *
from constants import *


def solve_d9fac9be(I):
    x1 = palette(I)
    x2 = objects(I, T, F, T)
    x3 = argmax(x2, size)
    x4 = color(x3)
    x5 = remove(ZERO, x1)
    x6 = other(x5, x4)
    O = canvas(x6, UNITY)
    return O
