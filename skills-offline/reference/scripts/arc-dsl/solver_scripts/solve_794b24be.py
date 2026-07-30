from dsl import *
from constants import *


def solve_794b24be(I):
    x1 = ofcolor(I, ONE)
    x2 = size(x1)
    x3 = decrement(x2)
    x4 = canvas(ZERO, THREE_BY_THREE)
    x5 = tojvec(x3)
    x6 = connect(ORIGIN, x5)
    x7 = equality(x2, FOUR)
    x8 = insert(UNITY, x6)
    x9 = branch(x7, x8, x6)
    O = fill(x4, TWO, x9)
    return O
