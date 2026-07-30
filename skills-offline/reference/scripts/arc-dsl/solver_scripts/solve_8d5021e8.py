from dsl import *
from constants import *


def solve_8d5021e8(I):
    x1 = vmirror(I)
    x2 = hconcat(x1, I)
    x3 = hmirror(x2)
    x4 = vconcat(x2, x3)
    x5 = vconcat(x4, x2)
    O = hmirror(x5)
    return O
