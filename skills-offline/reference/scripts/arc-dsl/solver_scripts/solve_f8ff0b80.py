from dsl import *
from constants import *


def solve_f8ff0b80(I):
    x1 = objects(I, T, T, T)
    x2 = order(x1, size)
    x3 = apply(color, x2)
    x4 = rbind(canvas, UNITY)
    x5 = apply(x4, x3)
    x6 = merge(x5)
    O = hmirror(x6)
    return O
