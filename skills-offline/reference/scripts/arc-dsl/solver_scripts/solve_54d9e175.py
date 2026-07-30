from dsl import *
from constants import *


def solve_54d9e175(I):
    x1 = objects(I, T, F, T)
    x2 = sizefilter(x1, ONE)
    x3 = compose(neighbors, center)
    x4 = fork(recolor, color, x3)
    x5 = mapply(x4, x2)
    x6 = paint(I, x5)
    x7 = replace(x6, ONE, SIX)
    x8 = replace(x7, TWO, SEVEN)
    x9 = replace(x8, THREE, EIGHT)
    O = replace(x9, FOUR, NINE)
    return O
