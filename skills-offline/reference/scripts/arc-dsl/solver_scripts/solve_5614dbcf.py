from dsl import *
from constants import *


def solve_5614dbcf(I):
    x1 = replace(I, FIVE, ZERO)
    O = downscale(x1, THREE)
    return O
