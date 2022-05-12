"""
DblQuaDot
=====

Provides
  1. 2 qubit system given by "Nature volume 569, pages532–536(2019)"
  2. Basic noise model introduced into the DQD (double quantum dot) system

"""

import numpy as np
from scipy.linalg import expm
import math

Ez = 3.933e10
dEz = 1.326e7
Omega = 410000.0
J = 1.59e6

dEz_tilde = np.sqrt(J**2 + dEz**2)

f_1u = (dEz_tilde+J)/2
f_1d = (dEz_tilde-J)/2
f_2u = (-dEz_tilde+J)/2
f_2d = (-dEz_tilde-J)/2

T_pi_2 = 1.0/(4.0*Omega)

alpha = (dEz + np.sqrt(J**2 + dEz**2)) / np.sqrt((dEz + np.sqrt(J**2 + dEz**2))**2 + J**2)
beta = J / np.sqrt((dEz + np.sqrt(J**2 + dEz**2))**2 + J**2)

QUASI_STATIC = True
STOCHASTIC = False

'''
Basic 2 qubit gate operations
'''

# swap the operation from Q1(Q2) to Q2(Q1)
def swap(m):
    s = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    return np.dot(np.dot(s, m), s)


'''
Lab frame Hamiltonian
'''
# a: amplitude ;
# f: excitation frequency ;
# t: time ;
# p: phase offset

# ac ESR magnetic field
def esr(a, f, t, p):
    return a * np.exp(-1j*(2*np.pi*f*t + p))

# Lab frame Hamiltonian (in diagonalized basis)
def h_lab(a, f, t, p):
    b = esr(a, f, t, p)
    return 1/2 * 2*np.pi * np.array([[0, b, b, 0],
                                     [np.conj(b), dEz_tilde-J, 0, b],
                                     [np.conj(b), 0, -dEz_tilde-J, b],
                                     [0, np.conj(b), np.conj(b), 0]])

# RWA transformation matrix (r @ U_rwa = U_lab)
def r(t):
    return np.diag([1, np.exp(1j*f_1d*2*np.pi*t), np.exp(-1j*f_1u*2*np.pi*t), 1])


'''
RWA frame Hamiltonian
'''
# a: esr pulse amplitude ;
# p: phase offset

# perfect gates
def h_rwa1_1d(a, p):
    return 1/2*2*np.pi*a*np.array([[0, 0, 0, 0], [0, 0, 0, np.exp(-1j*p)], [0, 0, 0, 0], [0, np.exp(1j*p), 0, 0]])

def h_rwa1_1u(a, p):
    return 1/2*2*np.pi*a*np.array([[0, 0, np.exp(-1j*p), 0], [0, 0, 0, 0], [np.exp(1j*p), 0, 0, 0], [0, 0, 0, 0]])


def h_rwa1_2d(a, p):
    return 1/2*2*np.pi*a*np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, np.exp(-1j*p)], [0, 0, np.exp(1j*p), 0]])

def h_rwa1_2u(a, p):
    return 1/2*2*np.pi*a*np.array([[0, np.exp(-1j*p), 0, 0], [np.exp(1j*p), 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

# crosstalk error terms
def h_rwa2_1d(a, t, p):
    return 1/2*2*np.pi*a*np.array([[0, 0, np.exp(1j*(J*2*np.pi*t - p)), 0], [0, 0, 0, 0],
                                   [np.exp(-1j*(J*2*np.pi*t - p)), 0, 0, 0], [0, 0, 0, 0]])

def h_rwa2_1u(a, t, p):
    return 1/2*2*np.pi*a*np.array([[0, 0, 0, 0], [0, 0, 0,  np.exp(-1j*(J*2*np.pi*t + p))],
                                   [0, 0, 0, 0], [0, np.exp(1j*(J*2*np.pi*t + p)), 0, 0]])

def h_rwa2_2d(a, t, p):
    return 1/2*2*np.pi*a*np.array([[0, np.exp(1j*(J*2*np.pi*t - p)), 0, 0], [np.exp(-1j*(J*2*np.pi*t - p)), 0, 0, 0],
                                   [0, 0, 0, 0], [0, 0, 0, 0]])

def h_rwa2_2u(a, t, p):
    return 1/2*2*np.pi*a*np.array([[0, 0, 0, 0], [0, 0, 0, 0],
                                   [0, 0, 0, np.exp(-1j*(J*2*np.pi*t + p))], [0, 0, np.exp(1j*(J*2*np.pi*t + p)), 0]])


'''
External error model
'''

# Energy fluctuation error (Gaussian fluctuations on diagonal elements)
# df_uu/ud/du/dd: fluctuations on uu/ud_tilde/du_tilde/dd under RWA state
def dh_e(df_uu, df_ud, df_du, df_dd):
    return 2*np.pi*np.array([[df_uu, 0, 0, 0],
                             [0, df_ud, 0, 0],
                             [0, 0, df_du, 0],
                             [0, 0, 0, df_dd]])

# Energy fluctuation error (Gaussian fluctuations on diagonal elements)
# df_uu/ud/du/dd: fluctuations on uu/ud/du/dd state
# t: gate time
def dh_e_tilde(df_uu, df_ud, df_du, df_dd, t):
    return 2*np.pi*np.array([[df_uu, 0, 0, 0],
                             [0, alpha**2*df_ud + beta**2*df_du,
                              np.exp(1j*(dEz_tilde-J)*t)*alpha*beta*(df_du-df_ud), 0],
                             [0, np.exp(-1j*(dEz_tilde-J)*t)*alpha*beta*(df_du-df_ud),
                              alpha**2*df_du + beta**2*df_ud, 0],
                             [0, 0, 0, df_dd]])

def dh_e_approx(df_uu, df_ud, df_du, df_dd):
    return 2*np.pi*np.array([[df_uu, 0, 0, 0],
                             [0, alpha**2*df_ud + beta**2*df_du, alpha*beta*(df_du-df_ud), 0],
                             [0, alpha*beta*(df_du-df_ud), alpha**2*df_du + beta**2*df_ud, 0],
                             [0, 0, 0, df_dd]])


'''
ESR pulses and primitive gates
'''

# combine perfect and cross error
def h_rwa_1d(a, t, p, noise, crosstalk=False):
    if crosstalk:
        return h_rwa1_1d(a, p) + h_rwa2_1d(a, t, p) + dh_e(noise[0], noise[1], noise[2], noise[3])
    else:
        return h_rwa1_1d(a, p) + dh_e(noise[0], noise[1], noise[2], noise[3])

def h_rwa_1u(a, t, p, noise, crosstalk=False):
    if crosstalk:
        return h_rwa1_1u(a, p) + h_rwa2_1u(a, t, p) + dh_e(noise[0], noise[1], noise[2], noise[3])
    else:
        return h_rwa1_1u(a, p) + dh_e(noise[0], noise[1], noise[2], noise[3])

def h_rwa_2d(a, t, p, noise, crosstalk=False):
    if crosstalk:
        return h_rwa1_2d(a, p) + h_rwa2_2d(a, t, p) + dh_e(noise[0], noise[1], noise[2], noise[3])
    else:
        return h_rwa1_2d(a, p) + dh_e(noise[0], noise[1], noise[2], noise[3])

def h_rwa_2u(a, t, p, noise, crosstalk=False):
    if crosstalk:
        return h_rwa1_2u(a, p) + h_rwa2_2u(a, t, p) + dh_e(noise[0], noise[1], noise[2], noise[3])
    else:
        return h_rwa1_2u(a, p) + dh_e(noise[0], noise[1], noise[2], noise[3])

# generate microwave pulse according to given index and noise type
# k: pulse index (0/1/2/3 for ESR frequency f_1u/f_1d/f_2u/f_2d)
# t_total: pulse time ;
# a: amplitude ;
# p: current phase record (based on applied pulses) ;
# delta: total time slice ;
# noise: an 4-elements array with fluctuations for 4 states ;
# noise_type: QUASI_STATIC or STOCHASTIC ;
# sgn: sign for exponential propagator. pi/2 (sgn=1) or -pi/2 (sgn=-1) pulse.
# -pi/2 pulse can be realized in experiment by adding a pi phase to ESR field.
def pulse_generate(k, t_total, a, p, delta, noise, noise_type=QUASI_STATIC, sgn=1, crosstalk=False):
    m = np.identity(4)
    if crosstalk:   # include crosstalk error
        t_slice = np.linspace(0, t_total, delta + 1)
        if noise_type:    # quasi-static noise
            dh_static = dh_e(noise[0], noise[1], noise[2], noise[3])
            if k == 0:
                for t in t_slice[1:]:
                    h = h_rwa1_1u(a, p[2] - p[0]) + h_rwa2_1u(a, t - (t_slice[1] / 2), p[2] - p[0]) + dh_static
                    m = np.dot(expm(-1j * sgn * h * t_slice[1]), m)
            elif k == 1:
                for t in t_slice[1:]:
                    h = h_rwa1_1d(a, p[3] - p[1]) + h_rwa2_1d(a, t - (t_slice[1] / 2), p[3] - p[1]) + dh_static
                    m = np.dot(expm(-1j * sgn * h * t_slice[1]), m)
            elif k == 2:
                for t in t_slice[1:]:
                    h = h_rwa1_2u(a, p[1] - p[0]) + h_rwa2_2u(a, t - (t_slice[1] / 2), p[1] - p[0]) + dh_static
                    m = np.dot(expm(-1j * sgn * h * t_slice[1]), m)
            elif k == 3:
                for t in t_slice[1:]:
                    h = h_rwa1_2d(a, p[3] - p[2]) + h_rwa2_2d(a, t - (t_slice[1] / 2), p[3] - p[2]) + dh_static
                    m = np.dot(expm(-1j * sgn * h * t_slice[1]), m)
        else:   # stochastic noise  # TODO: This is not really stochastic noise. No random fluctuations in dh_e here.
            if k == 0:              # TODO: Need to define a new randomized dh.
                for t in t_slice[1:]:
                    m = np.dot(expm(-1j * sgn * h_rwa_1u(a, t - (t_slice[1] / 2), p[2] - p[0], noise) * t_slice[1]), m)
            elif k == 1:
                for t in t_slice[1:]:
                    m = np.dot(expm(-1j * sgn * h_rwa_1d(a, t - (t_slice[1] / 2), p[3] - p[1], noise) * t_slice[1]), m)
            elif k == 2:
                for t in t_slice[1:]:
                    m = np.dot(expm(-1j * sgn * h_rwa_2u(a, t - (t_slice[1] / 2), p[1] - p[0], noise) * t_slice[1]), m)
            elif k == 3:
                for t in t_slice[1:]:
                    m = np.dot(expm(-1j * sgn * h_rwa_2d(a, t - (t_slice[1] / 2), p[3] - p[2], noise) * t_slice[1]), m)
    else:   # no crosstalk error
        if noise_type:    # quasi-static noise
            dh_static = dh_e(noise[0], noise[1], noise[2], noise[3])
            if k == 0:
                h = h_rwa1_1u(a, p[2] - p[0]) + dh_static
                m = expm(-1j * sgn * h * t_total)
            elif k == 1:
                h = h_rwa1_1d(a, p[3] - p[1]) + dh_static
                m = expm(-1j * sgn * h * t_total)
            elif k == 2:
                h = h_rwa1_2u(a, p[1] - p[0]) + dh_static
                m = expm(-1j * sgn * h * t_total)
            elif k == 3:
                h = h_rwa1_2d(a, p[3] - p[2]) + dh_static
                m = expm(-1j * sgn * h * t_total)
        else:   # stochastic noise  # TODO: This is not really stochastic noise. No random fluctuations in dh_e here.
            t_slice = np.linspace(0, t_total, delta + 1)
            if k == 0:
                for t in t_slice[1:]:
                    m = np.dot(expm(-1j * sgn * h_rwa_1u(a, t - (t_slice[1] / 2), p[2] - p[0], noise) * t_slice[1]), m)
            elif k == 1:
                for t in t_slice[1:]:
                    m = np.dot(expm(-1j * sgn * h_rwa_1d(a, t - (t_slice[1] / 2), p[3] - p[1], noise) * t_slice[1]), m)
            elif k == 2:
                for t in t_slice[1:]:
                    m = np.dot(expm(-1j * sgn * h_rwa_2u(a, t - (t_slice[1] / 2), p[1] - p[0], noise) * t_slice[1]), m)
            elif k == 3:
                for t in t_slice[1:]:
                    m = np.dot(expm(-1j * sgn * h_rwa_2d(a, t - (t_slice[1] / 2), p[3] - p[2], noise) * t_slice[1]), m)
    return m

# phase record for crosstalk error correction algorithm (obtain p_err by simulation)
def phase_rec(k, p):
    p_err = 0.03167654250993053 * np.pi  # phase error is given by simulation
    if k == 0:      # 1u
        p[1] += p_err
        p[3] -= p_err
    elif k == 1:    # 1d
        p[0] -= p_err
        p[2] += p_err
    elif k == 2:    # 2u
        p[2] += p_err
        p[3] -= p_err
    elif k == 3:    # 2d
        p[0] -= p_err
        p[1] += p_err

# v_z gates implemented by phase record
def v_z(k, p):
    if math.floor(k / 11):  # key = 11, 12, 13
        phase = (k - 10) * np.pi / 4
        p[0] += phase
        p[1] -= phase
        p[2] += phase
        p[3] -= phase
    else:                   # key = 8, 9, 10
        phase = (k - 7) * np.pi / 4
        p[0] += phase
        p[1] += phase
        p[2] -= phase
        p[3] -= phase

# generate gate (composed of two pulses or single v_z gates only) given prim_key
# prim_key is structured as follows:
# 'X(pi/2)' on Q2         = 0
# 'X(pi/2)', Q1           = 1
# 'X(pi/2)+CROT' on Q2    = 2
# 'X(pi/2)+CROT' on Q1    = 3
# 'Z(pi/2)+CROT' on Q2    = 4
# 'Z(pi/2)+CROT' on Q1    = 5
# 'CROT' on Q2            = 6
# 'CROT' on Q1            = 7
# 'Zv(pi/2)' on Q1        = 8
# 'Zv(pi)' on Q1          = 9
# 'Zv(3pi/2)' on Q1       = 10
# 'Zv(pi/2)' on Q2        = 11
# 'Zv(pi)' on Q2          = 12
# 'Zv(3pi/2)' on Q2       = 13
def get_gates(p, prim_key, a=Omega, delta=1001, t_total=T_pi_2, noise=None, noise_type=QUASI_STATIC, crosstalk=False):
    if noise is None:
        noise = [0, 0, 0, 0]
    if prim_key < 8:
        m1 = np.identity(4)
        m2 = np.identity(4)
        if prim_key == 0:
            m1 = pulse_generate(2, t_total, a, p, delta, noise, noise_type=noise_type, sgn=1, crosstalk=crosstalk)
            if crosstalk:
                phase_rec(2, p)
            m2 = pulse_generate(3, t_total, a, p, delta, noise, noise_type=noise_type, sgn=1, crosstalk=crosstalk)
            if crosstalk:
                phase_rec(3, p)
        elif prim_key == 1:
            m1 = pulse_generate(0, t_total, a, p, delta, noise, noise_type=noise_type, sgn=1, crosstalk=crosstalk)
            if crosstalk:
                phase_rec(0, p)
            m2 = pulse_generate(1, t_total, a, p, delta, noise, noise_type=noise_type, sgn=1, crosstalk=crosstalk)
            if crosstalk:
                phase_rec(1, p)
        elif prim_key == 2:
            m1 = pulse_generate(2, t_total, a, p, delta, noise, noise_type=noise_type, sgn=1, crosstalk=crosstalk)
            if crosstalk:
                phase_rec(2, p)
            m2 = pulse_generate(3, t_total, a, p, delta, noise, noise_type=noise_type, sgn=-1, crosstalk=crosstalk)
            if crosstalk:
                phase_rec(3, p)
        elif prim_key == 3:
            m1 = pulse_generate(0, t_total, a, p, delta, noise, noise_type=noise_type, sgn=1, crosstalk=crosstalk)
            if crosstalk:
                phase_rec(0, p)
            m2 = pulse_generate(1, t_total, a, p, delta, noise, noise_type=noise_type, sgn=-1, crosstalk=crosstalk)
            if crosstalk:
                phase_rec(1, p)
        elif prim_key == 4:
            m1 = pulse_generate(2, t_total, a, p, delta, noise, noise_type=noise_type, sgn=1, crosstalk=crosstalk)
            if crosstalk:
                phase_rec(2, p)
            m2 = pulse_generate(2, t_total, a, p, delta, noise, noise_type=noise_type, sgn=1, crosstalk=crosstalk)
            if crosstalk:
                phase_rec(2, p)
        elif prim_key == 5:
            m1 = pulse_generate(0, t_total, a, p, delta, noise, noise_type=noise_type, sgn=1, crosstalk=crosstalk)
            if crosstalk:
                phase_rec(0, p)
            m2 = pulse_generate(0, t_total, a, p, delta, noise, noise_type=noise_type, sgn=1, crosstalk=crosstalk)
            if crosstalk:
                phase_rec(0, p)
        elif prim_key == 6:
            m1 = pulse_generate(3, t_total, a, p, delta, noise, noise_type=noise_type, sgn=-1, crosstalk=crosstalk)
            if crosstalk:
                phase_rec(3, p)
            m2 = pulse_generate(3, t_total, a, p, delta, noise, noise_type=noise_type, sgn=-1, crosstalk=crosstalk)
            if crosstalk:
                phase_rec(3, p)
        elif prim_key == 7:
            m1 = pulse_generate(1, t_total, a, p, delta, noise, noise_type=noise_type, sgn=-1, crosstalk=crosstalk)
            if crosstalk:
                phase_rec(1, p)
            m2 = pulse_generate(1, t_total, a, p, delta, noise, noise_type=noise_type, sgn=-1, crosstalk=crosstalk)
            if crosstalk:
                phase_rec(1, p)
        return np.dot(m2, m1)
    else:
        v_z(prim_key, p)
        return np.identity(4)
