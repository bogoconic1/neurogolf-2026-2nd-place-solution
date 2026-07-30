from dsl import *
from constants import *


def solve_09629e4f(I):
    x1 = objects(I, F, T, T)
    x2 = argmin(x1, numcolors)
    x3 = normalize(x2)
    x4 = upscale(x3, FOUR)
    x5 = paint(I, x4)
    x6 = ofcolor(I, FIVE)
    O = fill(x5, FIVE, x6)
    return O
