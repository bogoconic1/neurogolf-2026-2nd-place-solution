from dsl import *
from constants import *


def solve_e98196ab(I):
    x1 = tophalf(I)
    x2 = bottomhalf(I)
    x3 = objects(x1, T, F, T)
    x4 = merge(x3)
    O = paint(x2, x4)
    return O
