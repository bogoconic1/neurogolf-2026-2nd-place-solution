from dsl import *
from constants import *


def solve_ea32f347(I):
    x1 = objects(I, T, F, T)
    x2 = replace(I, FIVE, FOUR)
    x3 = argmin(x1, size)
    x4 = argmax(x1, size)
    x5 = fill(x2, ONE, x4)
    O = fill(x5, TWO, x3)
    return O
