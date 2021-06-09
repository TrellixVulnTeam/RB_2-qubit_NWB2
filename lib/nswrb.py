"""
NSWRB
=====

Provides
  1. Based on 2 qubit system given by "Nature volume 569, pages532–536(2019)"
  2. Crosstalk error corrected by stark shift energy compensation
  3. Execute accelerated RB via time sliced Hamiltonian array

"""

from lib.twoqrb import *
import random

'''
tilde frame Hamiltonian with ac magnetic field off
'''
Hd = 1/2 * 2*np.pi * np.array([[0, 0, 0, 0],
                              [0, dEz_tilde-J, 0, 0],
                              [0, 0, -dEz_tilde-J, 0],
                              [0, 0, 0, 0]])

'''
ac magnetic field Hamiltonian with on-resonance rotating frame R = diag{exp(-ift), 1, 1, e(ift)}
'''
Xd = 1 / 2 * 2 * np.pi * np.array([[0, Omega, Omega, 0],
                                   [Omega, 0, 0, Omega],
                                   [Omega, 0, 0, Omega],
                                   [0, Omega, Omega, 0]])

'''
Crosstalk error correction with ac stark shift energy compensation
'''
# Return: phase accumulation angular frequencies for 4 levels (uu, ud, du, dd)
def ac_stark_frequency():
    x1 = 1/2 * 2 * np.pi * Omega * np.identity(4)

    F = np.diag([-1, 0, 0, 1])  # transform to rotating frame with R = diag{exp(-ift), 1, 1, e(ift)}
    f = np.array([f_1u, f_1d, f_2u, f_2d]) * 2 * np.pi

    d1 = np.empty([4, 4])

    for ii in range(4):
        hdd = np.sort(np.diag(Hd + (f[ii] + 1) * F))
        ix = np.argsort(np.diag(Hd + (f[ii] + 1) * F))
        iy = np.argsort(ix)
        w, v = np.linalg.eig(Hd + 1 * Xd + (f[ii] + 1) * F)
        ed = np.sort(w)
        d1[:, ii] = ed[iy] - hdd[iy]

    dd = d1 + np.array([[1, 0, 1, 0], [0, 1, -1, 0], [-1, 0, 0, 1], [0, -1, 0, -1]]) * np.diag(x1)
    phaseoffset = np.array([dd[0] - dd[2], dd[1] - dd[3], dd[0] - dd[1], dd[2] - dd[3]])
    return phaseoffset

# Return: phase accumulation for 4 levels (1u, 1d, 2u, 2d) from given ac stark shift frequencies f and time T
def ac_stark_modulation(f, t):
    return np.exp(1j * f * t)

def phase_rec_stark_shift(k, p_ac, p_rec):
    p_rec *= p_ac[:, k]

def v_z_4ac_pulse(k, p):
    if math.floor(k / 11):  # key = 11, 12, 13
        phase = (k - 10) * np.pi / 4
        p[2] *= np.exp(1j * 2 * phase)
        p[3] *= np.exp(1j * 2 * phase)
    else:                   # key = 8, 9, 10
        phase = (k - 7) * np.pi / 4
        p[0] *= np.exp(1j * 2 * phase)
        p[1] *= np.exp(1j * 2 * phase)


# generate gate waveform(composed of two pulses or single v_z gates only) given prim_key
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
def gate_waveform(prim_key, p_ac, p, dt, t_pi_2):
    t_slice = round(t_pi_2 / dt)
    waveform = np.zeros((4, 2 * t_slice))
    if prim_key == 0:
        waveform[2, :t_slice] = p[2]
        phase_rec_stark_shift(2, p_ac, p)
        waveform[3, t_slice:] = p[3]
        phase_rec_stark_shift(3, p_ac, p)
    elif prim_key == 1:
        waveform[0, :t_slice] = p[0]
        phase_rec_stark_shift(0, p_ac, p)
        waveform[1, t_slice:] = p[1]
        phase_rec_stark_shift(1, p_ac, p)
    elif prim_key == 2:
        waveform[2, :t_slice] = p[2]
        phase_rec_stark_shift(2, p_ac, p)
        waveform[3, t_slice:] = -1 * p[3]
        phase_rec_stark_shift(3, p_ac, p)
    elif prim_key == 3:
        waveform[0, :t_slice] = p[0]
        phase_rec_stark_shift(0, p_ac, p)
        waveform[1, t_slice:] = -1 * p[1]
        phase_rec_stark_shift(1, p_ac, p)
    elif prim_key == 4:
        waveform[2, :t_slice] = p[2]
        phase_rec_stark_shift(2, p_ac, p)
        waveform[2, :t_slice] = p[2]
        phase_rec_stark_shift(2, p_ac, p)
    elif prim_key == 5:
        waveform[0, :t_slice] = p[0]
        phase_rec_stark_shift(0, p_ac, p)
        waveform[0, :t_slice] = p[0]
        phase_rec_stark_shift(0, p_ac, p)
    elif prim_key == 6:
        waveform[3, t_slice:] = -1 * p[3]
        phase_rec_stark_shift(3, p_ac, p)
        waveform[3, t_slice:] = -1 * p[3]
        phase_rec_stark_shift(3, p_ac, p)
    elif prim_key == 7:
        waveform[1, t_slice:] = -1 * p[1]
        phase_rec_stark_shift(1, p_ac, p)
        waveform[1, t_slice:] = -1 * p[1]
        phase_rec_stark_shift(1, p_ac, p)
    return waveform


def cliff_waveform(keys, p_ac, p, dt, t_pi_2):
    wav = np.empty((4, 0))
    for i in reversed(range(len(keys))):
        if keys[i] < 8: # some real pulses occur
            a = gate_waveform(keys[i], p_ac, p, dt, t_pi_2)
            wav = np.append(wav, a, axis=1)
        else:   # virtual z's
            v_z_4ac_pulse(keys[i], p)
    return wav

# l: array contains gate lengths that is a data point (should do measurement) max(l) = cliff waveform length
def generate_rand_cliff_waveform(l, dt):
    for i in range(l):
        random.randint(0, 11520)




