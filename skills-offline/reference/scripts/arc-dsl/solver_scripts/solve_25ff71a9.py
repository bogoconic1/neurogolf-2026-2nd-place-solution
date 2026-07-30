from dsl import *
from constants import *


def solve_25ff71a9(I):
    x1 = objects(I, T, T, T)
    x2 = first(x1)
    O = move(I, x2, DOWN)
    return O
