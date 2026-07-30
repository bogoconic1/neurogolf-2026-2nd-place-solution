from dsl import *
from constants import *


def solve_017c7c7b(I):
    x1 = tophalf(I)
    x2 = bottomhalf(I)
    x3 = equality(x1, x2)
    x4 = crop(I, TWO_BY_ZERO, THREE_BY_THREE)
    x5 = branch(x3, x2, x4)
    x6 = vconcat(I, x5)
    O = replace(x6, ONE, TWO)
    return O
