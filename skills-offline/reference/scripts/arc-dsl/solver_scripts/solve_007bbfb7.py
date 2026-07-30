from dsl import *
from constants import *


def solve_007bbfb7(I):
    x1 = hupscale(I, THREE)
    x2 = vupscale(x1, THREE)
    x3 = hconcat(I, I)
    x4 = hconcat(x3, I)
    x5 = vconcat(x4, x4)
    x6 = vconcat(x5, x4)
    O = cellwise(x2, x6, ZERO)
    return O
