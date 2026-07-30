from dsl import *
from constants import *


def solve_ac0a08a4(I):
    x1 = colorcount(I, ZERO)
    x2 = subtract(NINE, x1)
    O = upscale(I, x2)
    return O
