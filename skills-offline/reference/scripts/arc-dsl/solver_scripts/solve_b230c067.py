from dsl import *
from constants import *


def solve_b230c067(I):
    x1 = objects(I, T, T, T)
    x2 = totuple(x1)
    x3 = apply(normalize, x2)
    x4 = leastcommon(x3)
    x5 = matcher(normalize, x4)
    x6 = extract(x1, x5)
    x7 = replace(I, EIGHT, ONE)
    O = fill(x7, TWO, x6)
    return O
