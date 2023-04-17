import qutip as qt
import numpy as np

import qiskit.quantum_info as qi
from qiskit.extensions import UnitaryGate

rng = np.random.default_rng()

def gen_haar_np():
    U6 = qt.rand_unitary_haar(6)
    U8 = np.zeros((8, 8), dtype=np.complex128)
    U8[0, 0] = 1
    U8[7, 7] = 1
    for i in range(1, 7):
        for j in range(1, 7):
            U8[i, j] = U6[i - 1, j - 1]
    return U8



def haar_qk():
    mat = gen_haar_np()
    return UnitaryGate(mat)

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


def get_domain_walls(bitstring):
    result = 0
    prev_bit = bitstring[0]
    for bit in bitstring:
        if bit != prev_bit:
            result = result + 1
        prev_bit = bit
    # print(bitstring, result)
    return result
    
class RCDM(object):
    def __init__(self, num, p = 0.):
        self.p = p
        self.num = num
        self.periodic_boundary = False
        self.random_init_state = True
        if self.random_init_state:
            state = qi.random_statevector(2 ** num)
        else:
            state = np.zeros(2 ** num)
            # state[0] = 1 / 2 ** 0.5
            # state[1] = 1 / 2 ** 0.5
            state[1] = 1
        data = qi.DensityMatrix(state)
        # data = qi.random_density_matrix(2 ** num)
        self.state = data

    def density(self):
        density = 0
        for i in range(2 ** self.num):
            bitstring = "{0:b}".format(i)
            if len(bitstring) < self.num:
                bitstring = '0' + bitstring
            w = self.state.data[i][i]
            w = np.real(w)
            density += w * get_domain_walls(bitstring)
        return density

    def evolve_random(self, op, qs):
        if rng.uniform(0, 1) < self.p:
            self.state = self.state.evolve(op, qs)

    def evolve_random_unitary(self, mod):
        i = mod
        while i + 2 < self.num:
            self.state = self.state.evolve(haar(), [i, i + 1, i + 2])
            i = i + 3
        if self.periodic_boundary:
            if i + 2 < self.num + mod:
                self.state = self.state.evolve(haar(), list(np.mod([i, i + 1, i + 2], self.num)))

    def evolve_probabilistic_measurement(self, mod):
        i = mod
        nops = self.num // 3
        ops = 0
        space = rng.integers(0, 3)
        i = i + space
        while ops < nops and i + 1 < self.num:
            # p = rng.uniform(0, 1)
            if rng.binomial(1, self.p):
                self.state = self.state.evolve(zz_pm(), [i, i + 1])
            space = rng.integers(0, 3)
            i = i + 2 + space
            ops += 1

        if ops < nops and self.periodic_boundary:
            # space = rng.integers(0, 3)
            # i = i + space
            if i + 1 < self.num + mod:
                if rng.binomial(1, self.p):
                    self.state = self.state.evolve(zz_pm(), list(np.mod([i, i + 1], self.num)))
    
            

    def evolve(self, rounds):
        period = 100
        entropies = np.zeros(rounds // period + 1)
        entropies[0] = qi.entropy(self.state)
        domain_wall_densities = np.zeros(rounds // period + 1)
        domain_wall_densities[0] = self.density()
        for t in range(rounds):
            # print(round)
            self.evolve_random_unitary(t % 3)
            self.evolve_probabilistic_measurement(t % 3)
            if (t + 1) % period == 0:
                entropies[(t + 1) // period] = qi.entropy(self.state)
                domain_wall_densities[(t + 1) // period] = self.density()
        return entropies, domain_wall_densities
