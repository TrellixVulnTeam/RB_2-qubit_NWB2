from lib.twoqrb import *

Hd = 1/2 * 2*np.pi * np.array([[0, 0, 0, 0],
                              [0, dEz_tilde-J, 0, 0],
                              [0, 0, -dEz_tilde-J, 0],
                              [0, 0, 0, 0]])

Xd = 1/2 * 2*np.pi * np.array([[0, Omega, Omega, 0],
                              [Omega, 0, 0, Omega],
                              [Omega, 0, 0, Omega],
                              [0, Omega, Omega, 0]])

x1 = 1/2 * 2*np.pi * Omega * np.identity(4)

F = np.diag([-1, 0, 0, 1])
f = np.array([f_1u, f_1d, f_2u, f_2d]) * 2 * np.pi

d1 = np.empty([4, 4])


for ii in range(4):
    hdd = np.sort(np.diag(Hd + (f[ii] + 1) * F))

    ix = np.argsort(np.diag(Hd + (f[ii] + 1) * F))
    iy = np.argsort(ix)
    w, v = np.linalg.eig(Hd + 1 * Xd + (f[ii] + 1) * F)
    ed = np.sort(w)
    d1[:, ii] = ed[iy] - hdd[iy]

# print(d1)
dd = d1 + np.array([[1, 0, 1, 0], [0, 1, -1, 0], [-1, 0, 0, 1], [0, -1, 0, -1]]) * np.diag(x1)
# print(np.array([[1, 0, 1, 0], [0, 1, -1, 0], [-1, 0, 0, 1], [0, -1, 0, -1]]) * np.diag(x1))
phaseoffset = np.array([dd[0]-dd[2], dd[1]-dd[3], dd[0]-dd[1], dd[2]-dd[3]])

modulation = np.exp(-1j*dd*T_pi_2)
# with np.printoptions(precision=3, suppress=True):
#     print(modulation)

def h_lab1(a, f, t, p):  # ideal lab frame Hamiltonian
    b = esr(a, f, t, p)
    return 1/2 * 2*np.pi * np.array([[0, 0, b, 0],                    # TODO: Need change elements for different pulses
                                     [0, dEz_tilde-J, 0, 0],
                                     [np.conj(b), 0, -dEz_tilde-J, 0],
                                     [0, 0, 0, 0]])

def h_lab2(a, f, t, p):  # real lab frame Hamiltonian
    b = esr(a, f, t, p)
    return 1/2 * 2*np.pi * np.array([[0, b, b, 0],
                                     [np.conj(b), dEz_tilde-J, 0, b],
                                     [np.conj(b), 0, -dEz_tilde-J, b],
                                     [0, np.conj(b), np.conj(b), 0]])


h_rwa1 = 1/2 * 2*np.pi * np.array([[-2*f_1u, Omega, Omega, 0],
                                   [Omega, dEz_tilde-J, 0, Omega],
                                   [Omega, 0, -dEz_tilde-J, Omega],
                                   [0, Omega, Omega, 2*f_1u]])

def h_rwa2(a, f, t, p):
    b = esr(a, f, t, p)
    return 1/2 * 2*np.pi * np.array([[0, 0, b*np.exp(1j*2*np.pi*f_1u*t), 0],
                                     [0, 0, 0, b*np.exp(1j*2*np.pi*f_1d*t)],
                                     [np.conj(b)*np.exp(-1j*2*np.pi*f_1u*t), 0, 0, 0],
                                     [0, np.conj(b)*np.exp(-1j*2*np.pi*f_1d*t), 0, 0]])


m_ideal = np.identity(4)
m_lab = np.identity(4)
m_rwa1 = np.identity(4)
m_rwa2 = np.identity(4)
dt = 5e-11
t_slice = round(T_pi_2 / dt)
time_list = dt * np.array(range(1, t_slice+1))
time_list2 = dt * np.array(range(t_slice+1, 2 * t_slice+1))

# p = [0, 0, 0, 0]    # phase accumulation for 4 on-resonance (1u, 1d, 2u, 2d)
# t_slice = np.linspace(0, T_pi_2, delta + 1)
for t in time_list:
    h1 = h_lab1(Omega, f_1u, t, 0)
    m_ideal = np.dot(expm(-1j * 1 * h1 * dt), m_ideal)

    h2 = h_lab2(Omega, f_1u, t, 0)
    m_lab = np.dot(expm(-1j * 1 * h2 * dt), m_lab)

    # m_rwa1 = np.dot(expm(-1j * 1 * h_rwa1 * t_slice[1]), m_rwa1)
    #
    # h3 = h_rwa2(Omega, f_1u, t, 0)
    # m_rwa2 = np.dot(expm(-1j * 1 * h3 * t_slice[1]), m_rwa2)

# t_slice2 = np.linspace(T_pi_2, 2 * T_pi_2, delta + 1)
# for t in time_list2:
#     h1 = h_lab1(Omega, f_2d, t, 0)
#     m_ideal = np.dot(expm(-1j * 1 * h1 * dt), m_ideal)
#
#     h2 = h_lab2(Omega, f_2d, t, 0)
#     m_lab = np.dot(expm(-1j * 1 * h2 * dt), m_lab)

print(m_lab)
# R2 = np.diag([np.exp(-1j*2*np.pi*f_1u*T_pi_2), 1, 1, np.exp(1j*2*np.pi*f_1u*T_pi_2)])
# m_rwa1 = R2 @ m_rwa1
mod2_1u = np.diag([np.exp(1j*(dd[0][0])*T_pi_2), np.exp(1j*dd[1][0]*T_pi_2), np.exp(1j*dd[2][0]*T_pi_2), np.exp(1j*(dd[3][0])*T_pi_2)])
# m_rwa1_ = mod2_1u @ m_rwa1
m_lab_ = mod2_1u @ m_lab

# print(modulation)
# print(mod2_1u)
R3 = np.diag([1, np.exp(-1j*2*np.pi*f_1d*T_pi_2), np.exp(1j*2*np.pi*f_1u*T_pi_2), 1])   # TODO: Need change T here
# print(m_rwa2)
# m_rwa2 = R3 @ m_rwa2
p_err = 0.03167654250993053 * np.pi
mod3_1u = np.diag([1, np.exp(-1j*p_err), 1, np.exp(1j*p_err)])
# m_rwa2_ = mod3_1u @ m_rwa2
m_lab__ = mod3_1u @ m_lab

U_1u = 1/np.sqrt(2) * np.array([[1, 0, -1j, 0], [0, np.sqrt(2), 0, 0], [-1j, 0, 1, 0], [0, 0, 0, np.sqrt(2)]])
U_1d = 1/np.sqrt(2) * np.array([[np.sqrt(2), 0, 0, 0], [0, 1, 0, -1j], [0, 0, np.sqrt(2), 0], [0, -1j, 0, 1]])
U_2u = 1/np.sqrt(2) * np.array([[1, -1j, 0, 0], [-1j, 1, 0, 0], [0, 0, np.sqrt(2), 0], [0, 0, 0, np.sqrt(2)]])
U_2d = 1/np.sqrt(2) * np.array([[np.sqrt(2), 0, 0, 0], [0, np.sqrt(2), 0, 0], [0, 0, 1, -1j], [0, 0, -1j, 1]])
# print(U_1u)

with np.printoptions(precision=5, suppress=True):
    # print(gate_fidelity(m_ideal, m_rwa1))
    # print(gate_fidelity(m_ideal, m_rwa1_))
    # print(gate_fidelity(m_ideal, m_rwa2))
    # print(gate_fidelity(m_ideal, m_rwa2_))
    print("Uncorrected: ", gate_fidelity(m_ideal, m_lab))
    print("Paper's method: ", gate_fidelity(m_ideal, m_lab_))
    print("My method: ", gate_fidelity(m_ideal, m_lab__))
    # print(gate_fidelity(R3 @ U_2d @ U_2u, m_lab))
    print("Compare with ideal unitary gate.")
    print("Paper's method: ", gate_fidelity(R3 @ U_1u, m_lab_))
    print("My method: ", gate_fidelity(R3 @ U_1u, m_lab__))
    print("Ideal Hamiltonian", gate_fidelity(R3 @ U_1u, m_ideal))




