import qecc as q
from lib.oneqrb import *

Cliff_decompose_1q = [[0, 1],                # X, -X
                      [0, 0],                # X^2
                      [3, 0, 0, 2],          # -Z, X^2, Z
                      [0, 2, 2, 0],          # X, Z^2, X
                      [0, 3, 0, 2],          # X, -Z, X, Z
                      [0, 2, 0, 3],          # X, Z, X, -Z
                      [1, 3, 0, 2],          # -X, -Z, X, Z
                      [1, 2, 0, 3],          # -X, Z, X, -Z
                      [3, 0, 2, 0],          # -Z, X, Z, X
                      [3, 0, 2, 1],          # -Z, X, Z, -X
                      [2, 0, 3, 0],          # Z, X, -Z, X
                      [2, 0, 3, 1],          # Z, X, -Z, -X
                      [3, 0, 2, 0, 3],       # -Z, X, Z, X, -Z
                      [2, 1, 3, 1, 2],       # Z, -X, -Z, -X, Z
                      [0, 2, 1],             # X, Z, -X
                      [0, 3, 1],             # X, -Z, -X
                      [1, 2, 2, 1, 3],       # -X, Z^2, -X, -Z
                      [1, 3, 3, 1, 2],       # -X, -Z^2, -X, Z
                      [0, 3, 0],             # X, -Z, X
                      [0, 2, 0],             # X, Z, X
                      [3, 0, 2, 0, 2],       # -Z, X, Z, X, Z
                      [3, 0, 2, 1, 3],       # -Z, X, Z, -X, -Z
                      [0, 0, 2],             # X^2, Z
                      [1, 1, 3]]             # -X^2, -Z


X = 1/np.sqrt(2) * np.array([[1, -1j],
                             [-1j, 1]])
X_minus = 1/np.sqrt(2) * np.array([[1, 1j],
                                   [1j, 1]])
Z = 1/np.sqrt(2) * np.array([[1-1j, 0],
                             [0, 1+1j]])
Z_minus = 1/np.sqrt(2) * np.array([[1+1j, 0],
                                   [0, 1-1j]])

prim_gates = [X, X_minus, Z, Z_minus]

Cliff_1 = []
for keys in Cliff_decompose_1q:
    M = np.identity(2)
    for j in reversed(range(len(keys))):
        M = prim_gates[keys[j]] @ M
    print(M)
    Cliff_1.append(M)

with open('Cliff1_unitary.pkl', 'wb') as f:
    pickle.dump(Cliff_1, f)

with open("Cliff1_unitary.pkl", "rb") as f1:
    Cliff1_decompose = pickle.load(f1)
f1.close()

it = q.clifford_group(1, consider_phases=True)

for i in range(24):
    c = next(it).as_unitary()
    for a in Cliff1_decompose:
        if is_inverse_1q(a.conj().T, c):
            print(str(i) + "True")

