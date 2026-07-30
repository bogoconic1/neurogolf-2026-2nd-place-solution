from dsl import *
from constants import *


def solve_85c4e7cd(I):
    x1 = objects(I, T, F, F)
    x2 = compose(invert, size)
    x3 = order(x1, size)
    x4 = order(x1, x2)
    x5 = apply(color, x4)
    x6 = mpapply(recolor, x5, x3)
    O = paint(I, x6)
    return O
