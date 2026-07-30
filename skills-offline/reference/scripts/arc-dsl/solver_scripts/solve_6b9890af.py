from dsl import *
from constants import *


def solve_6b9890af(I):
    x1 = objects(I, T, T, T)
    x2 = ofcolor(I, TWO)
    x3 = argmin(x1, size)
    x4 = subgrid(x2, I)
    x5 = width(x4)
    x6 = divide(x5, THREE)
    x7 = upscale(x3, x6)
    x8 = normalize(x7)
    x9 = shift(x8, UNITY)
    O = paint(x4, x9)
    return O
