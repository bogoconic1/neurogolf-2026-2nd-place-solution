from dsl import *
from constants import *


def solve_5117e062(I):
    x1 = objects(I, F, T, T)
    x2 = matcher(numcolors, TWO)
    x3 = extract(x1, x2)
    x4 = subgrid(x3, I)
    x5 = mostcolor(x3)
    O = replace(x4, EIGHT, x5)
    return O
