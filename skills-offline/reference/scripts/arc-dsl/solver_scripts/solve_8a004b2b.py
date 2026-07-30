from dsl import *
from constants import *


def solve_8a004b2b(I):
    x1 = objects(I, F, T, T)
    x2 = ofcolor(I, FOUR)
    x3 = subgrid(x2, I)
    x4 = argmax(x1, lowermost)
    x5 = normalize(x4)
    x6 = replace(x3, FOUR, ZERO)
    x7 = objects(x6, T, F, T)
    x8 = merge(x7)
    x9 = width(x8)
    x10 = ulcorner(x8)
    x11 = width(x4)
    x12 = divide(x9, x11)
    x13 = upscale(x5, x12)
    x14 = shift(x13, x10)
    O = paint(x3, x14)
    return O
