from dsl import *
from constants import *


def solve_694f12f3(I):
    x1 = objects(I, T, F, F)
    x2 = colorfilter(x1, FOUR)
    x3 = compose(backdrop, inbox)
    x4 = argmin(x2, size)
    x5 = argmax(x2, size)
    x6 = x3(x4)
    x7 = x3(x5)
    x8 = fill(I, ONE, x6)
    O = fill(x8, TWO, x7)
    return O
