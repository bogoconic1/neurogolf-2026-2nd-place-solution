from dsl import *
from constants import *


def solve_445eab21(I):
    x1 = objects(I, T, F, T)
    x2 = fork(multiply, height, width)
    x3 = argmax(x1, x2)
    x4 = color(x3)
    O = canvas(x4, TWO_BY_TWO)
    return O
