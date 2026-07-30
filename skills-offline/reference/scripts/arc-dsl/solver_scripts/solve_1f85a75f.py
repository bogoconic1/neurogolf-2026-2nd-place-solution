from dsl import *
from constants import *


def solve_1f85a75f(I):
    x1 = objects(I, T, T, T)
    x2 = argmax(x1, size)
    O = subgrid(x2, I)
    return O
