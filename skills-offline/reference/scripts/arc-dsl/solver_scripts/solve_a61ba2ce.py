from dsl import *
from constants import *


def solve_a61ba2ce(I):
    x1 = objects(I, T, F, T)
    x2 = lbind(index, I)
    x3 = matcher(x2, ZERO)
    x4 = lbind(extract, x1)
    x5 = rbind(subgrid, I)
    x6 = lbind(compose, x3)
    x7 = chain(x5, x4, x6)
    x8 = x7(ulcorner)
    x9 = x7(urcorner)
    x10 = x7(llcorner)
    x11 = x7(lrcorner)
    x12 = hconcat(x11, x10)
    x13 = hconcat(x9, x8)
    O = vconcat(x12, x13)
    return O
