from dsl import *
from constants import *


def solve_239be575(I):
    x1 = objects(I, F, T, T)
    x2 = lbind(contained, TWO)
    x3 = compose(x2, palette)
    x4 = sfilter(x1, x3)
    x5 = size(x4)
    x6 = greater(x5, ONE)
    x7 = branch(x6, ZERO, EIGHT)
    O = canvas(x7, UNITY)
    return O
