from dsl import *
from constants import *


def solve_8f2ea7aa(I):
    x1 = objects(I, T, F, T)
    x2 = merge(x1)
    x3 = subgrid(x2, I)
    x4 = upscale(x3, THREE)
    x5 = hconcat(x3, x3)
    x6 = hconcat(x5, x3)
    x7 = vconcat(x6, x6)
    x8 = vconcat(x7, x6)
    O = cellwise(x4, x8, ZERO)
    return O
