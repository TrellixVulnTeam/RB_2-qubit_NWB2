import qecc as q
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from lib.twoqrb import *

Hd = 1/2 * 2*np.pi * np.array([[0, 0, 0, 0],
                              [0, dEz_tilde-J, 0, 0],
                              [0, 0, -dEz_tilde-J, 0],
                              [0, 0, 0, 0]])

Xd = 1/2 * 2*np.pi * np.array([[0, Omega, Omega, 0],
                              [Omega, 0, 0, Omega],
                              [Omega, 0, 0, Omega],
                              [0, Omega, Omega, 0]])

x1 = np.pi * Omega * np.identity(4)

F = np.diag([-1, 0, 0, 1])
f = (np.array([f_1u, f_1d, f_2u, f_2d]) - Ez) * 2 * np.pi

d1 = np.empty([4, 4])


for ii in range(4):
    hdd = np.sort(np.diag(Hd + (f[ii] + 1) * F))

    ix = np.argsort(np.diag(Hd + (f[ii] + 1) * F))
    iy = np.argsort(ix)
    w, v = np.linalg.eig(Hd + 1 * Xd + (f[ii] + 1) * F)
    ed = np.sort(w)
    d1[:, ii] = ed[iy] - hdd[iy]


dd = d1 + np.array([[1, 0, 1, 0], [0, 1, -1, 0], [-1, 0, 0, 1], [0, -1, 0, -1]]) * np.diag(x1)
# print(dd)
phaseoffset = np.array([dd[0]-dd[2], dd[1]-dd[3], dd[0]-dd[1], dd[2]-dd[3]])

modulation = phaseoffset * T_pi_2 / np.pi
# with np.printoptions(precision=3, suppress=True):
#     print(modulation)

f_1u = (dEz_tilde+J)/2
f_1d = (dEz_tilde-J)/2
f_2u = (-dEz_tilde+J)/2
f_2d = (-dEz_tilde-J)/2

def h_lab1(a, f, t, p):  # ideal lab frame Hamiltonian for 1u pulse
    b = esr(a, f, t, p)
    return 1/2 * 2*np.pi * np.array([[0, 0, b, 0],
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

def prob_project(m):
    return np.diag(m) * np.diag(m). conj()


m_ideal = np.identity(4)
m_lab = np.identity(4)
m_rwa1 = np.identity(4)
m_rwa2 = np.identity(4)

delta = 1000

t_slice = np.linspace(0, T_pi_2, delta + 1)
# for t in t_slice[1:]:
#     h1 = h_lab1(Omega, f_1u, t, 0)
#     m_ideal = np.dot(expm(-1j * 1 * h1 * t_slice[1]), m_ideal)
#
#     h2 = h_lab2(Omega, f_1u, t, 0)
#     m_lab = np.dot(expm(-1j * 1 * h2 * t_slice[1]), m_lab)


t_slice2 = np.linspace(T_pi_2, 2 * T_pi_2, delta + 1)
for t in t_slice2[1:]:
    h1 = h_lab1(Omega, f_1d, t, 0)
    m_ideal = np.dot(expm(-1j * 1 * h1 * t_slice[1]), m_ideal)

    h2 = h_lab2(Omega, f_1d, t, 0)
    m_lab = np.dot(expm(-1j * 1 * h2 * t_slice[1]), m_lab)

print(m_lab)

R2 = np.diag([np.exp(-1j*2*np.pi*f_1u*T_pi_2), 1, 1, np.exp(1j*2*np.pi*f_1u*T_pi_2)])

R3 = np.diag([1, np.exp(-1j*2*np.pi*f_1d*T_pi_2), np.exp(1j*2*np.pi*f_1u*T_pi_2), 1])
m_rwa2 = R3 @ m_rwa2

U_1u = 1/np.sqrt(2) * np.array([[1, 0, -1j, 0], [0, np.sqrt(2), 0, 0], [-1j, 0, 1, 0], [0, 0, 0, np.sqrt(2)]])
U_1d = 1/np.sqrt(2) * np.array([[np.sqrt(2), 0, 0, 0], [0, 1, 0, -1j], [0, 0, np.sqrt(2), 0], [0, -1j, 0, 1]])
U_2u = 1/np.sqrt(2) * np.array([[1, -1j, 0, 0], [-1j, 1, 0, 0], [0, 0, np.sqrt(2), 0], [0, 0, 0, np.sqrt(2)]])
U_2d = 1/np.sqrt(2) * np.array([[np.sqrt(2), 0, 0, 0], [0, np.sqrt(2), 0, 0], [0, 0, 1, -1j], [0, 0, -1j, 1]])

R3_2 = np.diag([1, np.exp(-1j*2*np.pi*f_1d*2*T_pi_2), np.exp(1j*2*np.pi*f_1u*2*T_pi_2), 1])

A = R3_2 @ U_1d @ U_1u
B = m_lab
with np.printoptions(precision=5, suppress=True):
    # print(gate_fidelity(m_ideal, m_rwa1))
    # print(gate_fidelity(m_ideal, m_rwa1_))
    # print(gate_fidelity(m_ideal, m_rwa2))
    # print(gate_fidelity(m_ideal, m_rwa2_))
    # print(gate_fidelity(m_ideal, m_lab))
    # print(gate_fidelity(m_ideal, m_lab_))
    # print(gate_fidelity(R3 @ U_1u, m_ideal))
    # print(gate_fidelity(R3 @ U_1u, m_lab))
    # print(gate_fidelity(R3 @ U_1u, m_lab_))
    # print(A @ A.conj().T)
    # print(gate_fidelity(R3_2 @ U_1d @ R3.conj().T, m_lab))
    print(gate_fidelity(R3_2 @ U_1d @ R3.conj().T, m_lab))
    # print(prob_project(A))
    # print(prob_project(B))




