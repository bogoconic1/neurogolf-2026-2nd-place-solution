from dsl import *
from constants import *


def solve_9565186b(I):
    x1 = shape(I)
    x2 = objects(I, T, F, F)
    x3 = argmax(x2, size)
    x4 = canvas(FIVE, x1)
    O = paint(x4, x3)
    return O
