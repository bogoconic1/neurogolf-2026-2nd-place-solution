from dsl import *
from constants import *


def solve_44f52bb0(I):
    x1 = vmirror(I)
    x2 = equality(x1, I)
    x3 = branch(x2, ONE, SEVEN)
    O = canvas(x3, UNITY)
    return O
