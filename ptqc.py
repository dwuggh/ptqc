import qutip as qt
import matplotlib as plt
import numpy as np

import qiskit as qk
import qiskit.circuit as qc
import qiskit.quantum_info as qi
import random

def gen_haar_np():
    U6 = qt.rand_unitary_haar(6)
    U8 = np.zeros((8, 8), dtype=np.complex128)
    U8[0, 0] = 1
    U8[7, 7] = 1
    for i in range(1, 7):
        for j in range(1, 7):
            U8[i, j] = U6[i - 1, j - 1]
    return U8

def haar():
    mat = gen_haar_np()
    return qi.Operator(mat)

def zz_pm():
    op1 = np.mat([
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        ])
    # op2 = np.mat([
    #     [0, 0, 0, 0],
    #     [0, 1, 0, 0],
    #     [0, 0, 1, 0],
    #     [0, 0, 0, 0],
    #     ])

    # x = np.mat([
    #     [0, 0, 1, 0],
    #     [0, 0, 0, 1],
    #     [1, 0, 0, 0],
    #     [0, 1, 0, 0],
    #     ])
    # op2 = np.matmul(x, op2)

    op2 = np.mat([
        [0, 0, 1, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        ])
    return qi.Kraus([op1, op2])

    
class RCDM(object):
    def __init__(self, num, p = 0):
        self.p = p
        self.num = num
        data = qi.random_density_matrix(2 ** num)
        self.state = data

    def evolve_random(self, op, qs):
        if random.uniform(0, 1) < self.p:
            self.state = self.state.evolve(op, qs)

    def evolve_random_unitary(self, mod):
        i = mod
        while i + 2 < self.num:
            self.state = self.state.evolve(haar(), [i, i + 1, i + 2])
            i = i + 3

    def evolve_probabilistic_measurement(self, mod):
        i = mod
        while i + 1 < self.num:
            p = random.uniform(0, 1)
            if i < self.p:
                self.state = self.state.evolve(zz_pm(), [i, i + 1])

            space = random.randint(0, 2)
            i = i + 2 + space
    
            

    def evolve(self, rounds):
        entropies = []
        for t in range(rounds):
            # print(round)
            self.evolve_random_unitary(t % 3)
            # self.state = self.state.evolve(haar(), [0, 1, 2])
            # self.state = self.state.evolve(haar(), [3, 4, 5])
            self.evolve_random(zz_pm(), [1, 2])
            self.evolve_random(zz_pm(), [4, 5])

            self.state = self.state.evolve(haar(), [1, 2, 3])
            self.state = self.state.evolve(haar(), [4, 5, 6])
            self.evolve_random(zz_pm(), [2, 3])
            self.evolve_random(zz_pm(), [4, 5])

            self.state = self.state.evolve(haar(), [2, 3, 4])
            self.evolve_random(zz_pm(), [1, 2])
            self.evolve_random(zz_pm(), [5, 6])

            entropies.append(qi.entropy(self.state))
        return entropies

# a = qi.DensityMatrix(qi.Statevector([1, 0]))

# a.evolve
