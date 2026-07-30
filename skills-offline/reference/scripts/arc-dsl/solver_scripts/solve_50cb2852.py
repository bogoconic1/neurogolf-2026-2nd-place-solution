from dsl import *
from constants import *


def solve_50cb2852(I):
    x1 = objects(I, T, F, T)
    x2 = compose(backdrop, inbox)
    x3 = mapply(x2, x1)
    O = fill(I, EIGHT, x3)
    return O
