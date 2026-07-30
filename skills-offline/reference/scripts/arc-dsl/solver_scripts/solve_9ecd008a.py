from dsl import *
from constants import *


def solve_9ecd008a(I):
    x1 = vmirror(I)
    x2 = ofcolor(I, ZERO)
    O = subgrid(x2, x1)
    return O
