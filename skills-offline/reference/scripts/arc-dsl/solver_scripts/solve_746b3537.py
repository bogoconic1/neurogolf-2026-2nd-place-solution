from dsl import *
from constants import *


def solve_746b3537(I):
    x1 = chain(size, dedupe, first)
    x2 = x1(I)
    x3 = equality(x2, ONE)
    x4 = branch(x3, dmirror, identity)
    x5 = x4(I)
    x6 = objects(x5, T, F, F)
    x7 = order(x6, leftmost)
    x8 = apply(color, x7)
    x9 = repeat(x8, ONE)
    O = x4(x9)
    return O
