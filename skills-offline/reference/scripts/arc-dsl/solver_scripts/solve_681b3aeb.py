from dsl import *
from constants import *


def solve_681b3aeb(I):
    x1 = rot270(I)
    x2 = objects(x1, T, F, T)
    x3 = argmax(x2, size)
    x4 = argmin(x2, size)
    x5 = color(x4)
    x6 = canvas(x5, THREE_BY_THREE)
    x7 = normalize(x3)
    x8 = paint(x6, x7)
    O = rot90(x8)
    return O
