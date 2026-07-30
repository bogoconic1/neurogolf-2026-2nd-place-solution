from dsl import *
from constants import *


def solve_dc433765(I):
    x1 = ofcolor(I, THREE)
    x2 = ofcolor(I, FOUR)
    x3 = first(x1)
    x4 = first(x2)
    x5 = subtract(x4, x3)
    x6 = sign(x5)
    x7 = recolor(THREE, x1)
    O = move(I, x7, x6)
    return O
