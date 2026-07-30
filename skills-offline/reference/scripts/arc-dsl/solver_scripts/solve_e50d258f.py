from dsl import *
from constants import *


def solve_e50d258f(I):
    x1 = width(I)
    x2 = astuple(NINE, x1)
    x3 = canvas(ZERO, x2)
    x4 = vconcat(I, x3)
    x5 = objects(x4, F, F, T)
    x6 = rbind(colorcount, TWO)
    x7 = argmax(x5, x6)
    O = subgrid(x7, I)
    return O
