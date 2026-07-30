from dsl import *
from constants import *


def solve_bb43febb(I):
    x1 = objects(I, T, F, F)
    x2 = colorfilter(x1, FIVE)
    x3 = compose(backdrop, inbox)
    x4 = mapply(x3, x2)
    O = fill(I, TWO, x4)
    return O
