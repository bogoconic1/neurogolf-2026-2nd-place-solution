from dsl import *
from constants import *


def solve_54d82841(I):
    x1 = height(I)
    x2 = objects(I, T, F, T)
    x3 = compose(last, center)
    x4 = apply(x3, x2)
    x5 = decrement(x1)
    x6 = lbind(astuple, x5)
    x7 = apply(x6, x4)
    O = fill(I, FOUR, x7)
    return O
