from dsl import *
from constants import *


def solve_7f4411dc(I):
    x1 = leastcolor(I)
    x2 = ofcolor(I, x1)
    x3 = rbind(difference, x2)
    x4 = rbind(greater, TWO)
    x5 = chain(x4, size, x3)
    x6 = compose(x5, dneighbors)
    x7 = sfilter(x2, x6)
    O = fill(I, ZERO, x7)
    return O
