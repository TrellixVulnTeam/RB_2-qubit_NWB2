from lib.oneqrb import *

c = get_cliff_1q(10, delta_t=100, noise_type=HAMILTONIAN_NOISE, noise_angle=0)
c_ = Cliff_perfect_1q[10]

print(c)
print(c_)
print(gate_fidelity_1q(c, c_))
print(Cliff_decompose_1q[10])
c_test = Z_pi_2 @ X_pi_2 @ Z_pi_2minus @ X_pi_2
print(gate_fidelity_1q(c_test, c_))
