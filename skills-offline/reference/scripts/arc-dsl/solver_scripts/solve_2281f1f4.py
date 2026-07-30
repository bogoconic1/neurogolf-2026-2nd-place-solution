from dsl import *
from constants import *


def solve_2281f1f4(I):
    x1 = ofcolor(I, FIVE)
    x2 = product(x1, x1)
    x3 = power(first, TWO)
    x4 = power(last, TWO)
    x5 = fork(astuple, x3, x4)
    x6 = apply(x5, x2)
    x7 = urcorner(x1)
    x8 = remove(x7, x6)
    O = underfill(I, TWO, x8)
    return O
