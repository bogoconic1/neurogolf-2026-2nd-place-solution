from dsl import *
from constants import *


def solve_1c786137(I):
    x1 = objects(I, T, F, F)
    x2 = argmax(x1, height)
    x3 = subgrid(x2, I)
    O = trim(x3)
    return O
