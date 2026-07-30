from dsl import *
from constants import *


def solve_a1570a43(I):
    x1 = ofcolor(I, TWO)
    x2 = ofcolor(I, THREE)
    x3 = recolor(TWO, x1)
    x4 = ulcorner(x2)
    x5 = ulcorner(x1)
    x6 = subtract(x4, x5)
    x7 = increment(x6)
    O = move(I, x3, x7)
    return O
