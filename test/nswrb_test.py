from lib.nswrb import *
import random

offset_f = ac_stark_frequency()
# phase_comp = ac_stark_modulation(offset_f, T_pi_2)
phase_comp = np.ones((4, 4))
dt = 5e-9
f = np.array([[f_1u, f_1d, f_2u, f_2d]])
rho_0 = error_initial_state(0, 0, 0)
L = [1]

cliff_seq = [[12, 0, 10, 5, 11, 2, 8, 12, 7, 4, 9, 1, 9, 11]]
# cliff_seq = random.choices(Cliff_decompose, k=L[-1])
# cliff_seq = [[0]]
wav, tindex, p_rec = generate_cliff_waveform(cliff_seq, L, dt, phase_comp)
H_seq = waveform_2_H(wav, dt, f)
# print(wav)
rho_list, U = time_evolve_2(H_seq, dt, rho_0)
print(U)

inv = get_perfect_inverse_set(cliff_seq, L)
for i in range(len(L)):
    C = get_perfect_seq(cliff_seq[:L[i]])
    # print(L[i])
    # print(gate_fidelity(C @ inv[i], np.identity(4)))
    n = 2 * sum(i < 8 for i in sum(cliff_seq[:L[i]], []))
    print("n = ", n)
    print(gate_fidelity(r(n*T_pi_2) @ np.diag(p_rec[i]).conj() @ U, C))
# print(gate_fidelity(U @ inv[0], np.identity(4)))
#
#
# print(C)

