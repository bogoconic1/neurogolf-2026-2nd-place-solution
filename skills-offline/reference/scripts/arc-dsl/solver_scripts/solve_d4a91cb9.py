from dsl import *
from constants import *


def solve_d4a91cb9(I):
    x1 = ofcolor(I, EIGHT)
    x2 = ofcolor(I, TWO)
    x3 = first(x1)
    x4 = first(x2)
    x5 = last(x3)
    x6 = first(x4)
    x7 = astuple(x6, x5)
    x8 = connect(x7, x3)
    x9 = connect(x7, x4)
    x10 = combine(x8, x9)
    O = underfill(I, FOUR, x10)
    return O
