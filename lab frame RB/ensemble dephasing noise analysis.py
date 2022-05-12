from lib.nswrb import *
from qiskit import quantum_info
from qiskit.providers.aer.noise import depolarizing_error
from qiskit.quantum_info.operators.channel import ptm
import matplotlib.pyplot as plt
import copy
import scipy.stats as stats
import sys

'''
1. Assume that crosstalk-error is perfectly cancelled.
2. All Clifford gate are generated in theoretical form not in lab frame.
3. There is only ensemble-dephasing noise.
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
    m = 1 / N * m
    w, v = np.linalg.eig(m)
    return w

def pdf_delta_list(mid, sigma, deltas):
    res = [0] * len(deltas)
    d = stats.norm(mid, sigma)
    for i in range(len(res)):
        res[i] = d.pdf(deltas[i])
    return res


sigma_list = 1000 * np.array(range(1, 21))     # Unit in frequency (need to multiply 2pi)
sigma_max = sigma_list[-1]

n = 100
d = 4
N = 11520

delta_list = [-3 * sigma_max + 6 * x * sigma_max / n for x in range(n+1)]

# cliffords_ensemble_list[i][j] : i-th Clifford with noise parameter delta_list[j]
cliffords_ensemble_list = []
for i in range(N):
    cliffords_ensemble_list.append([])

# noise_ensemble_list[i][j] : right-noise gate on i-th Clifford with noise parameter delta_list[j]
noise_ensemble_list = []
for i in range(N):
    noise_ensemble_list.append([])

perfect_clifford_list = []
for i in range(len(Cliff_decompose)):
    perfect_clifford_list.append(get_perfect_cliff(Cliff_decompose[i]))

# F_ave = []  # average gate fidelity of right channel
# F_wallman = []

for delta in delta_list:
    print("delta_list #", delta_list.index(delta), "is working...")
    primitive_list = noisy_primitives_rwa_2q(delta)

    # get noisy clifford gate with noise parameter delta
    def get_cliff(keys):
        g_pf = np.identity(4)
        for i in reversed(range(len(keys))):
            g_pf = primitive_list[keys[i]] @ g_pf
        return g_pf

    for i in range(N):
        tex = ("Clifford # " + str(i) + " is working...")
        sys.stdout.write('\r' + tex)
        c = get_cliff(Cliff_decompose[i])
        cliffords_ensemble_list[i].append(c)
        noise_ensemble_list[i].append(perfect_clifford_list[i].conj().T @ c)
    sys.stdout.write('\r')
print("Done!")

f1 = open('cliffords_ensemble_noises_samples_2q.pkl', 'wb')
pickle.dump(cliffords_ensemble_list, f1)
f1.close()

f2 = open('ensemble_noises_samples_2q.pkl', 'wb')
pickle.dump(delta_list, f2)
f2.close()

print("Clifford ensemble noise samplings are ready!")

f_delta_list = [[0] * len(delta_list)] * N
for i in range(N):
    tex = ("Preparing average gate fidelity for Clifford # " + str(i) + "...")
    sys.stdout.write('\r' + tex)
    for j in range(len(delta_list)):
        ch = quantum_info.Operator(noise_ensemble_list[i][j])
        f_delta_list[i][j] = quantum_info.average_gate_fidelity(ch)
sys.stdout.write('\r')
print("Done!")

f3 = open('f_ave_ensemble_noises_samples_2q.pkl', 'wb')
pickle.dump(f_delta_list, f3)
f3.close()

fidelity_avg_hamiltonian = []
fidelity_thr_hamiltonian = []

for sigma in sigma_list:
    print("Counting F_avg for sigma = ", str(sigma), " ...")
    # average gate fidelity
    pdf = pdf_delta_list(0, sigma, delta_list)
    f_clifford = [0] * N
    for i in range(len(f_clifford)):
        tex = ("Counting f_avg for Clifford # " + str(i) + "...")
        sys.stdout.write('\r' + tex)
        f = 0
        for j in range(len(pdf)):
            f += pdf[j] * (6 * sigma_max / n) * f_delta_list[i][j]
        f_clifford[i] = f
    fidelity_avg_hamiltonian.append(np.mean(f_clifford))
    sys.stdout.write('\r')
    print("Done!")

    print("Counting F_wallman for sigma = ", str(sigma), " ...")
    # Wallman fidelity
    g_tilde_ptm_list = []
    for i in range(N):
        tex = ("Preparing g_tilde_ptm for Clifford # " + str(i) + "...")
        sys.stdout.write('\r' + tex)
        g_tilde_ptm_i = np.zeros((16, 16))
        for j in range(len(pdf)):
            g_tilde_ptm_i = g_tilde_ptm_i + pdf[j] * (6 * sigma_max / n) * \
                          ptm.PTM(quantum_info.Operator(cliffords_ensemble_list[i][j])).data
        g_tilde_ptm_list.append(g_tilde_ptm_i)
    sys.stdout.write('\r')

    m = np.zeros((256, 256))
    print("Solving eigenvalue problem...")
    for i in range(N):
        tex = ("Preparing m for Clifford # " + str(i) + "...")
        sys.stdout.write('\r' + tex)
        g_ptm = ptm.PTM(quantum_info.Operator(perfect_clifford_list[i])).data
        depol_ch_ptm = ptm.PTM(depolarizing_error(1, 2)).data
        g_u_ptm = g_ptm - depol_ch_ptm
        g_tilde_ptm = g_tilde_ptm_list[i]
        m = m + np.kron(g_u_ptm, g_tilde_ptm)
    m = 1 / N * m
    w, v = np.linalg.eig(m)
    sys.stdout.write('\r')
    print("Done!")

    p_wallman = np.real(np.amax(w))
    fidelity_thr_hamiltonian.append((p_wallman*(d-1)+1)/d)


f1 = open('thr_ensemble_sigma_list_2q.pkl', 'wb')
pickle.dump(sigma_list, f1)
f1.close()

f3 = open('thr_ensemble_sigma_f_hamiltonian_avg_2q.pkl', 'wb')
pickle.dump(fidelity_avg_hamiltonian, f3)
f3.close()

f4 = open('thr_ensemble_sigma_f_hamiltonian_rb_2q.pkl', 'wb')
pickle.dump(fidelity_thr_hamiltonian, f4)
f4.close()

plot2 = plt.figure(1)
plt.plot(sigma_list, fidelity_avg_hamiltonian, 'bo', markersize=2, label='Hamiltonian noise average')
plt.plot(sigma_list, fidelity_thr_hamiltonian, 'go', markersize=2, label='Hamiltonian noise RB theoretical')

plt.title('Average gate fidelity F with constant noise')
plt.xlabel("noise strength sigma (Hz)")
plt.ylabel("Clifford average gate fidelity")
plt.legend()
plt.show()
