from dsl import *
from constants import *


def solve_46f33fce(I):
    x1 = rot180(I)
    x2 = downscale(x1, TWO)
    x3 = rot180(x2)
    O = upscale(x3, FOUR)
    return O
