from dsl import *
from constants import *


def solve_662c240a(I):
    x1 = vsplit(I, THREE)
    x2 = fork(equality, dmirror, identity)
    x3 = compose(flip, x2)
    O = extract(x1, x3)
    return O
