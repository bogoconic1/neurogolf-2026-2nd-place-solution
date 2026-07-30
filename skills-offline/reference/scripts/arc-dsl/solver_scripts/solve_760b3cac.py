from dsl import *
from constants import *


def solve_760b3cac(I):
    x1 = ofcolor(I, FOUR)
    x2 = ofcolor(I, EIGHT)
    x3 = ulcorner(x1)
    x4 = index(I, x3)
    x5 = equality(x4, FOUR)
    x6 = branch(x5, NEG_ONE, ONE)
    x7 = multiply(x6, THREE)
    x8 = tojvec(x7)
    x9 = vmirror(x2)
    x10 = shift(x9, x8)
    O = fill(I, EIGHT, x10)
    return O
