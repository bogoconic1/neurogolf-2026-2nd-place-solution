from dsl import *
from constants import *


def solve_f76d97a5(I):
    x1 = palette(I)
    x2 = first(x1)
    x3 = last(x1)
    x4 = switch(I, x2, x3)
    O = replace(x4, FIVE, ZERO)
    return O
