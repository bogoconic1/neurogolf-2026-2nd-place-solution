from dsl import *
from constants import *


def solve_aabf363d(I):
    x1 = leastcolor(I)
    x2 = replace(I, x1, ZERO)
    x3 = leastcolor(x2)
    O = replace(x2, x3, x1)
    return O
