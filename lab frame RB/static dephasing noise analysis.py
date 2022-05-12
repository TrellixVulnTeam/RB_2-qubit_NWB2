from lib.nswrb import *
from qiskit import quantum_info
from qiskit.providers.aer.noise import depolarizing_error
from qiskit.quantum_info.operators.channel import ptm
import matplotlib.pyplot as plt
import copy
# from test.phase_correction import *
# TODO: test every pulses are the same as experimental pulses

'''
1. Assume that crosstalk-error is perfectly cancelled.
2. All Clifford gate are generated in theoretical form not in lab frame.
3. There is only constant-dephasing noise.
'''
I_1q = np.identity(2)
X_1q = np.array([[0, 1],
                 [1, 0]])
Y_1q = np.array([[0, -1j],
                 [1j, 0]])
Z_1q = np.array([[1, 0],
                 [0, -1]])

# generate 8 noisy pulses from given dephasing noise strength delta
def pauli_x_1q(angle):
    return np.cos(angle / 2) * I_1q - 1j * np.sin(angle / 2) * X_1q

def pauli_y_1q(angle):
    return np.cos(angle / 2) * I_1q - 1j * np.sin(angle / 2) * Y_1q

def pauli_z_1q(angle):
    return np.cos(angle / 2) * I_1q - 1j * np.sin(angle / 2) * Z_1q

def noisy_pi_2_unitary_1q(sgn_omega, sgn_delta, delta):
    delta_tilde = 2 * delta / Omega
    omega = np.sqrt(1 + delta_tilde**2)
    gamma = np.arcsin(delta_tilde / omega)

    g1 = pauli_y_1q(-sgn_omega * sgn_delta * gamma)
    g2 = pauli_x_1q(sgn_omega * np.pi / 2 * omega)
    g3 = pauli_y_1q(sgn_omega * sgn_delta * gamma)

    return g1 @ g2 @ g3

def noisy_pulses_rwa_2q(delta):
    delta_tilde = 2 * delta / Omega
    u1 = noisy_pi_2_unitary_1q(1, 1, delta)
    u2 = noisy_pi_2_unitary_1q(-1, 1, delta)
    u3 = noisy_pi_2_unitary_1q(1, -1, delta)
    u4 = noisy_pi_2_unitary_1q(-1, -1, delta)
    z1 = pauli_z_1q(np.pi / 2 * delta_tilde)
    z2 = pauli_z_1q(-np.pi / 2 * delta_tilde)

    u_2u = np.block([[u1, np.zeros((2, 2))],
                          [np.zeros((2, 2)), z2]])

    u_m_2u = np.block([[u2, np.zeros((2, 2))],
                          [np.zeros((2, 2)), z2]])

    u_2d = np.block([[z1, np.zeros((2, 2))],
                           [np.zeros((2, 2)), u3]])

    u_m_2d = np.block([[z1, np.zeros((2, 2))],
                           [np.zeros((2, 2)), u4]])

    u_1u = copy.deepcopy(u_2u)
    u_1u[:, [1, 2]] = u_1u[:, [2, 1]]
    u_1u[[1, 2], :] = u_1u[[2, 1], :]

    u_m_1u = copy.deepcopy(u_m_2u)
    u_m_1u[:, [1, 2]] = u_m_1u[:, [2, 1]]
    u_m_1u[[1, 2], :] = u_m_1u[[2, 1], :]

    u_1d = copy.deepcopy(u_2d)
    u_1d[:, [1, 2]] = u_1d[:, [2, 1]]
    u_1d[[1, 2], :] = u_1d[[2, 1], :]

    u_m_1d = copy.deepcopy(u_m_2d)
    u_m_1d[:, [1, 2]] = u_m_1d[:, [2, 1]]
    u_m_1d[[1, 2], :] = u_m_1d[[2, 1], :]

    return [u_1u, u_1d, u_2u, u_2d, u_m_1u, u_m_1d, u_m_2u, u_m_2d]

def noisy_primitives_rwa_2q(delta):
    pulse_list = noisy_pulses_rwa_2q(delta)
    p0 = pulse_list[3] @ pulse_list[2]
    p1 = pulse_list[1] @ pulse_list[0]
    p2 = pulse_list[7] @ pulse_list[2]
    p3 = pulse_list[5] @ pulse_list[0]
    p4 = pulse_list[2] @ pulse_list[2]
    p5 = pulse_list[0] @ pulse_list[0]
    p6 = pulse_list[7] @ pulse_list[7]
    p7 = pulse_list[5] @ pulse_list[5]
    p8 = Prim_perfect[8]
    p9 = Prim_perfect[9]
    p10 = Prim_perfect[10]
    p11 = Prim_perfect[11]
    p12 = Prim_perfect[12]
    p13 = Prim_perfect[13]
    p14 = Prim_perfect[14]

    return [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14]

def find_theoretical_p_2q(noisy_clifford, perfect_clifford):
    m = np.zeros((256, 256))
    for i in range(len(noisy_clifford)):
        g_ptm = ptm.PTM(quantum_info.Operator(perfect_clifford[i])).data
        depol_ch_ptm = ptm.PTM(depolarizing_error(1, 2)).data
        g_u_ptm = g_ptm - depol_ch_ptm
        g_tilde_ptm = ptm.PTM(quantum_info.Operator(noisy_clifford[i])).data
        m = m + np.kron(g_u_ptm, g_tilde_ptm)
    m = 1 / 11520 * m
    w, v = np.linalg.eig(m)
    return w


delta_list = 1000 * np.array(range(21))     # Unit in frequency (need to multiply 2pi)
# delta_list = [0, 2000, 4000]
F_ave = []  # average gate fidelity of right channel
F_wallman = []
d = 4

for delta in delta_list:
    print("delta = ", delta, "is working...")
    primitive_list = noisy_primitives_rwa_2q(delta)

    def get_cliff(keys):
        g_pf = np.identity(4)
        for i in reversed(range(len(keys))):
            g_pf = primitive_list[keys[i]] @ g_pf
        return g_pf

    noisy_clifford_list = []
    perfect_clifford_list = []
    for i in range(len(Cliff_decompose)):
        noisy_clifford_list.append(get_cliff(Cliff_decompose[i]))
        perfect_clifford_list.append(get_perfect_cliff(Cliff_decompose[i]))

    # extract right channel
    right_channel_list = []
    for i in range(len(noisy_clifford_list)):
        right_channel_list.append(perfect_clifford_list[i].conj().T @ noisy_clifford_list[i])
    # calculate average gate fidelity from right channel noise
    f_clifford = []
    for i in range(len(right_channel_list)):
        ch = quantum_info.Operator(right_channel_list[i])
        f = quantum_info.average_gate_fidelity(ch)
        f_clifford.append(f)
    F_ave.append(np.mean(f_clifford))

    # calculate Wallman RB decay parameter
    p_wallman = np.real(np.amax(find_theoretical_p_2q(noisy_clifford_list, perfect_clifford_list)))
    F_wallman.append((p_wallman*(d-1)+1)/d)
    print("Done!")

f1 = open('thr_const_delta_f_hamiltonian_avg_2q.pkl', 'wb')
pickle.dump(F_ave, f1)
f1.close()

f2 = open('thr_const_delta_f_hamiltonian_rb_2q.pkl', 'wb')
pickle.dump(F_wallman, f2)
f2.close()

f3 = open('thr_const_delta_list_2q.pkl', 'wb')
pickle.dump(delta_list, f3)
f3.close()


plot2 = plt.figure(1)
plt.plot(delta_list, F_ave, 'bo', markersize=2, label='Hamiltonian noise average')
plt.plot(delta_list, F_wallman, 'go', markersize=2, label='Hamiltonian noise RB theoretical')

plt.title('Two-qubit average gate fidelity F with constant noise')
plt.xlabel("noise strength delta (Hz)")
plt.ylabel("Clifford average gate fidelity")
plt.legend()
plt.show()

