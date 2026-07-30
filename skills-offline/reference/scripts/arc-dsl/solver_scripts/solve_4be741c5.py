from dsl import *
from constants import *


def solve_4be741c5(I):
    x1 = portrait(I)
    x2 = branch(x1, dmirror, identity)
    x3 = branch(x1, height, width)
    x4 = x3(I)
    x5 = astuple(ONE, x4)
    x6 = x2(I)
    x7 = crop(x6, ORIGIN, x5)
    x8 = apply(dedupe, x7)
    O = x2(x8)
    return O
