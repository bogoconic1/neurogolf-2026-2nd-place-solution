from dsl import *
from constants import *


def solve_a61f2674(I):
    x1 = objects(I, T, F, T)
    x2 = argmax(x1, size)
    x3 = argmin(x1, size)
    x4 = replace(I, FIVE, ZERO)
    x5 = recolor(ONE, x2)
    x6 = recolor(TWO, x3)
    x7 = combine(x5, x6)
    O = paint(x4, x7)
    return O
