from sys import argv
import numpy as np
from ptqc import *

if __name__ == '__main__':
    n = int(argv[1])
    p = float(argv[2])
    if argv.__len__() >= 4:
        rounds = int(argv[3])
    else:
        rounds = 20000
    ρ = RCDM(n, p)
    ent = ρ.evolve(rounds)
    np.savetxt("results/{}-{}.txt".format(n, p), ent, fmt="%.8f")
