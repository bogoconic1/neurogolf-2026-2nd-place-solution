from dsl import *
from constants import *


def solve_e3497940(I):
    x1 = lefthalf(I)
    x2 = righthalf(I)
    x3 = vmirror(x2)
    x4 = objects(x3, T, F, T)
    x5 = merge(x4)
    O = paint(x1, x5)
    return O
