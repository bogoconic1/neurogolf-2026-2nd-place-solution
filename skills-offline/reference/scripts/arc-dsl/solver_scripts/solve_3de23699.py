from dsl import *
from constants import *


def solve_3de23699(I):
    x1 = fgpartition(I)
    x2 = sizefilter(x1, FOUR)
    x3 = first(x2)
    x4 = difference(x1, x2)
    x5 = first(x4)
    x6 = color(x3)
    x7 = color(x5)
    x8 = subgrid(x3, I)
    x9 = trim(x8)
    O = replace(x9, x7, x6)
    return O
