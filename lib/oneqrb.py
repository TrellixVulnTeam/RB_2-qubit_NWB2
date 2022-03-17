"""
OneQubitRandomizedBenchmarking
=====

Provides
  1. Randomized benchmarking for 1 qubit system with noisy X rotation and perfect Z rotation

"""

import numpy as np
from scipy.linalg import expm
import pickle

CHANNEL_NOISE = True
HAMILTONIAN_NOISE = False

'''
Clifford decomposition
'''


'''
Basic 1 qubit gate operations
'''

# fidelity of two 1-qubit gates
def gate_fidelity_1q(m_exp, m):
    return (np.absolute(np.trace(np.dot(m_exp.conj().T, m))))**2/4

# check whether two gates m1 and m2 are the same
def is_inverse_1q(m1, m2):
    if np.allclose(np.absolute(np.trace(np.dot(m1, m2))), 2):
        return True
    else:
        return False

# prim_key is structured as follows:
# 'X(pi/2)'               = 0
# 'X(-pi/2)', Q1          = 1
# 'Z(pi/2)'               = 2
# 'Z(-pi/2)'              = 3


Cliff_decompose_1q = np.array([[0, 1],              # X, -X
                               [0, 0],              # X^2
                               [3, 0, 0, 2],        # -Z, X^2, Z
                               [0, 2, 2, 0],        # X, Z^2, X
                               [0, 3, 0, 2],        # X, -Z, X, Z
                               [0, 2, 0, 3],        # X, Z, X, -Z
                               [1, 3, 0, 2],        # -X, -Z, X, Z
                               [1, 2, 0, 3],        # -X, Z, X, -Z
                               [3, 0, 2, 0],        # -Z, X, Z, X
                               [3, 0, 2, 1],        # -Z, X, Z, -X
                               [2, 0, 3, 0],        # Z, X, -Z, X
                               [2, 0, 3, 1],        # Z, X, -Z, -X
                               [3, 0, 2, 0, 3],     # -Z, X, Z, X, -Z
                               [2, 1, 3, 1, 2],     # Z, -X, -Z, -X, Z
                               [0, 2, 1],           # X, Z, -X
                               [0, 3, 1],           # X, -Z, -X
                               [1, 2, 2, 1, 3],     # -X, Z^2, -X, -Z
                               [1, 3, 3, 1, 2],     # -X, -Z^2, -X, Z
                               [0, 3, 0],           # X, -Z, X
                               [0, 2, 0],           # X, Z, X
                               [3, 0, 2, 0, 2],     # -Z, X, Z, X, Z
                               [3, 0, 2, 1, 3],     # -Z, X, Z, -X, -Z
                               [0, 0, 2],           # X^2, Z
                               [1, 1, 3]])          # -X^2, -Z

X_pi_2 = 1/np.sqrt(2) * np.array([[1, -1j],
                             [-1j, 1]])
X_pi_2minus = 1/np.sqrt(2) * np.array([[1, 1j],
                                   [1j, 1]])
Z_pi_2 = 1/np.sqrt(2) * np.array([[1-1j, 0],
                             [0, 1+1j]])
Z_pi_2minus = 1/np.sqrt(2) * np.array([[1+1j, 0],
                                   [0, 1-1j]])

prim_1q = [X_pi_2, X_pi_2minus, Z_pi_2, Z_pi_2minus]

I_1q = np.identity(2)
X_1q = np.array([[0, 1],
                 [1, 0]])
Z_1q = np.array([[1, 0],
                 [0, -1]])

with open("../lib/1q_Cliff_perfect.pkl", "rb") as f1:
    Cliff_perfect_1q = pickle.load(f1)
f1.close()

def get_perfect_cliff(idx):
    m = Cliff_perfect_1q[idx]
    return m


# given a single Cliff element index (0~24) then return experimental gate of the Cliff element
# delta_t : dt for each X_pi_2 pulse
# noise_type: CHANNEL_NOISE or HAMILTONIAN_NOISE
# noise_angle: dephasing noise rotation angle.
#              If "noise_type" = "HAMILTONIAN_NOISE", the rotation angle will be divided into
def get_cliff_1q(idx, delta_t=1000, noise_type=CHANNEL_NOISE, noise_angle=0):
    if noise_type:   # dephasing noise is in the channel form and follows a perfect clifford gate
        m = get_perfect_cliff(idx)
        noise_ch = np.cos(noise_angle) * I_1q - 1j * np.sin(noise_angle) * Z_1q    # TODO: Channel noise is X right now
        return noise_ch @ m
    else:   # dephasing noise is in the Hamiltonian
        m = I_1q
        t_slice = np.linspace(0, np.pi/2, delta_t + 1)
        keys = Cliff_decompose_1q[idx]
        for i in reversed(range(len(keys))):
            if keys[i] == 0:     # X_pi_2
                x_pi2 = I_1q
                hx = 1/2 * X_1q
                hz = noise_angle/2/(np.pi/2) * 1/2 * Z_1q
                h = hx + hz
                for t in t_slice[1:]:
                    x_pi2 = np.dot(expm(-1j * h * t_slice[1]), x_pi2)
                m = x_pi2 @ m
            elif keys[i] == 1:   # -X_pi_2
                x_pi2m = I_1q
                hx = -1/2 * X_1q
                hz = noise_angle/2/(np.pi/2) * 1/2 * Z_1q
                h = hx + hz
                for t in t_slice[1:]:
                    x_pi2m = np.dot(expm(-1j * h * t_slice[1]), x_pi2m)
                m = x_pi2m @ m
            elif keys[i] == 2:   # Z_pi_2
                m = Z_pi_2 @ m
            elif keys[i] == 3:   # -Z_pi_2
                m = Z_pi_2minus @ m
        return m

# given a Cliff sequence (list with elements 0~23) then return list of all the ordered experimental Clifford gates
def get_seq_1q(idx_seq, noise_ang_seq, delta_t=1000, noise_type=CHANNEL_NOISE):
    if len(idx_seq) != len(noise_ang_seq):
        print("Clifford gate length does not match the noise length.")
        return None
    cliff_seq = []
    for i in range(len(idx_seq)):
        m = get_cliff_1q(idx_seq[i], delta_t=delta_t, noise_type=noise_type, noise_angle=noise_ang_seq[i])
        cliff_seq.append(m)
    return cliff_seq

def get_perfect_seq_1q(idx_seq):
    m = I_1q
    for i in range(len(idx_seq)):
        m = get_perfect_cliff(idx_seq[i]) @ m
    return m

# given Cliff sequence, return its perfect inverse gate
def get_seq_inverse(idx_seq):
    m = get_perfect_seq_1q(idx_seq)
    for i in range(len(Cliff_perfect_1q)):
        n = Cliff_perfect_1q[i]
        if is_inverse_1q(m, n):
            return n

# given a Cliff sequence (list with elements 0~23) then return list of all the ordered experimental Clifford gates
def get_seq_1q_h_diff_pulse_noise(idx_seq, noise_ang_seq, delta_t=1000):
    if len(idx_seq) != len(noise_ang_seq):
        print("Clifford gate length does not match the noise length.")
        return None
    cliff_seq = []
    for i in range(len(idx_seq)):
        m = get_cliff_1q_diff_pulse_noise(idx_seq[i], delta_t=delta_t, noise_angle=noise_ang_seq[i])
        cliff_seq.append(m)
    return cliff_seq

def get_cliff_1q_diff_pulse_noise(idx, delta_t=1000, noise_angle=np.array([0, 0])):
    m = I_1q
    t_slice = np.linspace(0, np.pi / 2, delta_t + 1)
    keys = Cliff_decompose_1q[idx]
    count = 0   # pulse number counting
    for i in reversed(range(len(keys))):
        if keys[i] == 0:  # X_pi_2
            x_pi2 = I_1q
            hx = 1 / 2 * X_1q
            hz = noise_angle[count] / (np.pi / 2) * 1 / 2 * Z_1q
            h = hx + hz
            count += 1
            for t in t_slice[1:]:
                x_pi2 = np.dot(expm(-1j * h * t_slice[1]), x_pi2)
            m = x_pi2 @ m
        elif keys[i] == 1:  # -X_pi_2
            x_pi2m = I_1q
            hx = -1 / 2 * X_1q
            hz = noise_angle[count] / (np.pi / 2) * 1 / 2 * Z_1q
            h = hx + hz
            count += 1
            for t in t_slice[1:]:
                x_pi2m = np.dot(expm(-1j * h * t_slice[1]), x_pi2m)
            m = x_pi2m @ m
        elif keys[i] == 2:  # Z_pi_2
            m = Z_pi_2 @ m
        elif keys[i] == 3:  # -Z_pi_2
            m = Z_pi_2minus @ m
    return m



