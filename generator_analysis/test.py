import qecc as q
import numpy as np
from lib.oneqrb import *

cliff = q.clifford_group(1)

X = 1/np.sqrt(2) * np.array([[1, -1j],
                             [-1j, 1]])
X_minus = 1/np.sqrt(2) * np.array([[1, -1j],
                             [-1j, 1]])
Z = 1/np.sqrt(2) * np.array([[1-1j, 0],
                             [0, 1+1j]])
Z_minus = 1/np.sqrt(2) * np.array([[1+1j, 0],
                                   [0, 1-1j]])

test = Z @ X @ X @ Z_minus
# test = Z @ X @ Z_minus @ X
# test = X_minus @ Z @ Z @ X_minus @ Z_minus
# test = Z_minus @ X_minus @ Z @ Z @ X_minus
print(test)

cliff = q.clifford_group(1, consider_phases=True)

for i in range(24):
    c = next(cliff).as_unitary()
    if is_inverse_1q(c.conj().T, test):
        print(c)
