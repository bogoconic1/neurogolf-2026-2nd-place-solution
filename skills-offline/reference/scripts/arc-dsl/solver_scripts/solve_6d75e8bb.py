from dsl import *
from constants import *


def solve_6d75e8bb(I):
    x1 = objects(I, T, F, T)
    x2 = first(x1)
    x3 = ulcorner(x2)
    x4 = subgrid(x2, I)
    x5 = replace(x4, ZERO, TWO)
    x6 = asobject(x5)
    x7 = shift(x6, x3)
    O = paint(I, x7)
    return O
