from dsl import *
from constants import *


def solve_0520fde7(I):
    x1 = vmirror(I)
    x2 = lefthalf(x1)
    x3 = righthalf(x1)
    x4 = vmirror(x3)
    x5 = cellwise(x2, x4, ZERO)
    O = replace(x5, ONE, TWO)
    return O
