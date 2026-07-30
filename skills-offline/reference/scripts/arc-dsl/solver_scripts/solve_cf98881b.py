from dsl import *
from constants import *


def solve_cf98881b(I):
    x1 = hsplit(I, THREE)
    x2 = first(x1)
    x3 = remove(x2, x1)
    x4 = first(x3)
    x5 = last(x3)
    x6 = ofcolor(x4, NINE)
    x7 = ofcolor(x2, FOUR)
    x8 = fill(x5, NINE, x6)
    O = fill(x8, FOUR, x7)
    return O
