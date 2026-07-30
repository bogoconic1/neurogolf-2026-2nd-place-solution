from dsl import *
from constants import *


def solve_27a28665(I):
    x1 = objects(I, T, F, F)
    x2 = valmax(x1, size)
    x3 = equality(x2, ONE)
    x4 = equality(x2, FOUR)
    x5 = equality(x2, FIVE)
    x6 = branch(x3, TWO, ONE)
    x7 = branch(x4, THREE, x6)
    x8 = branch(x5, SIX, x7)
    O = canvas(x8, UNITY)
    return O
