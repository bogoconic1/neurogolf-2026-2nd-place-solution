from dsl import *
from constants import *


def solve_d406998b(I):
    x1 = vmirror(I)
    x2 = ofcolor(x1, FIVE)
    x3 = compose(even, last)
    x4 = sfilter(x2, x3)
    x5 = fill(x1, THREE, x4)
    O = vmirror(x5)
    return O
