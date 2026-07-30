from dsl import *
from constants import *


def solve_8be77c9e(I):
    x1 = hmirror(I)
    O = vconcat(I, x1)
    return O
