"""
NSWRB
=====

Provides
  1. Based on 2 qubit system given by "Nature volume 569, pages532–536(2019)"
  2. Crosstalk error corrected by stark shift energy compensation
  3. Execute accelerated RB via time sliced Hamiltonian array

"""

from lib.twoqrb import *
import copy

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
# Return: phase accumulation angular frequencies for 4 energies (1u, 1d, 2u, 2d)
def ac_stark_frequency():
    x1 = 1/2 * 2 * np.pi * Omega * np.identity(4)

    F = np.diag([-1, 0, 0, 1])  # transform to rotating frame with R = diag{exp(-ift), 1, 1, e(ift)}
    f = np.array([f_1u, f_1d, f_2u, f_2d]) * 2 * np.pi

    d1 = np.empty([4, 4], dtype=np.complex)

    for ii in range(4):
        hdd = np.sort(np.diag(Hd + (f[ii] + 1) * F))
        ix = np.argsort(np.diag(Hd + (f[ii] + 1) * F))
        iy = np.argsort(ix)
        w, v = np.linalg.eig(Hd + 1 * Xd + (f[ii] + 1) * F)
        ed = np.sort(w)
        d1[:, ii] = ed[iy] - hdd[iy]

    dd = d1 + np.array([[1, 0, 1, 0], [0, 1, -1, 0], [-1, 0, 0, 1], [0, -1, 0, -1]]) * np.diag(x1)
    # phaseoffset = np.array([dd[0] - dd[2], dd[1] - dd[3], dd[0] - dd[1], dd[2] - dd[3]])
    return dd

# Return: phase accumulation for 4 levels (uu, ud, du, dd) from given ac stark shift frequencies f and time T
def ac_stark_modulation(f, t):
    return np.exp(-1j * f * t)

def phase_rec_stark_shift(k, p_ac, p_rec):
    p_rec *= p_ac[:, k]

def v_z_4ac_pulse(k, p):
    if math.floor(k / 11):  # key = 11, 12, 13
        phase = (k - 10) * np.pi / 4
        p[0] *= np.exp(1j * phase)
        p[1] *= np.exp(-1j * phase)
        p[2] *= np.exp(1j * phase)
        p[3] *= np.exp(-1j * phase)
    else:                   # key = 8, 9, 10
        phase = (k - 7) * np.pi / 4
        p[0] *= np.exp(1j * phase)
        p[1] *= np.exp(1j * phase)
        p[2] *= np.exp(-1j * phase)
        p[3] *= np.exp(-1j * phase)


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
    waveform = np.zeros((4, 2 * t_slice), dtype=np.complex)
    if prim_key == 0:
        waveform[2, :t_slice] = p[0]/p[1]
        p = p * p_ac[:, 2]
        waveform[3, t_slice:] = p[2]/p[3]
        p = p * p_ac[:, 3]
    elif prim_key == 1:
        waveform[0, :t_slice] = p[0]/p[2]
        p = p * p_ac[:, 0]
        waveform[1, t_slice:] = p[1]/p[3]
        p = p * p_ac[:, 1]
    elif prim_key == 2:
        waveform[2, :t_slice] = p[0]/p[1]
        p = p * p_ac[:, 2]
        waveform[3, t_slice:] = -1 * p[2]/p[3]
        p = p * p_ac[:, 3]
    elif prim_key == 3:
        waveform[0, :t_slice] = p[0]/p[2]
        p = p * p_ac[:, 0]
        waveform[1, t_slice:] = -1 * p[1]/p[3]
        p = p * p_ac[:, 1]
    elif prim_key == 4:
        waveform[2, :t_slice] = p[0]/p[1]
        p = p * p_ac[:, 2]
        waveform[2, t_slice:] = p[0]/p[1]
        p = p * p_ac[:, 2]
    elif prim_key == 5:
        waveform[0, :t_slice] = p[0]/p[2]
        p = p * p_ac[:, 0]
        waveform[0, t_slice:] = p[0]/p[2]
        p = p * p_ac[:, 0]
    elif prim_key == 6:
        waveform[3, :t_slice] = -1 * p[2]/p[3]
        p = p * p_ac[:, 3]
        waveform[3, t_slice:] = -1 * p[2]/p[3]
        p = p * p_ac[:, 3]
    elif prim_key == 7:
        waveform[1, :t_slice] = -1 * p[1]/p[3]
        p = p * p_ac[:, 1]
        waveform[1, t_slice:] = -1 * p[1]/p[3]
        p = p * p_ac[:, 1]
    p_new = p
    return waveform, p_new


def cliff_waveform(keys, p_ac, p, dt, t_pi_2):
    p_rec = p
    wav = np.empty((4, 0))
    for i in reversed(range(len(keys))):
        if keys[i] < 8:  # some real pulses occur
            # print("phase before: ", p_rec)
            a, p_new = gate_waveform(keys[i], p_ac, p_rec, dt, t_pi_2)
            p_rec = p_new
            wav = np.append(wav, a, axis=1)
            # print("waveform: ", a[:, 0], a[:, -1])
            # print("phase after: ", p_rec)
            # print("----------------------------")
        else:   # virtual z's
            # print("phase before: ", p_rec)
            v_z_4ac_pulse(keys[i], p_rec)
            # print("phase after: ", p_rec)
            # print("----------------------------")
    return wav, p_rec

# get list of gates that are the inverses of seq[0:l] where l is in the "data", the gate length should be a data point.
def get_perfect_inverse_set(seq, data):
    perfect_inverse_set = []
    if len(seq) != data[-1]:
        print("waveform sequence length does not match max gate length!")
        return None
    g = np.identity(4)
    for i in range(len(seq)):
        keys = seq[i]   # decomposition of i-th Cliff
        a = get_perfect_cliff(keys)
        g = a @ g
        if (i+1) in data:
            for j in range(len(Cliff_decompose)):
                b = get_perfect_cliff(Cliff_decompose[j])
                if is_inverse(b, g):
                    perfect_inverse_set.append(b)
    return perfect_inverse_set

# data: array contains gate lengths that is a data point (should do measurement) max(data) = cliff waveform length
def generate_cliff_waveform(seq, data, dt, phase_comp):
    waveform = np.empty((4, 0))
    p = [1, 1, 1, 1]
    data_tindex = []
    data_prec = []
    for i in range(len(seq)):
        keys = seq[i]
        a, p_new = cliff_waveform(keys, phase_comp, p, dt, T_pi_2)
        waveform = np.append(waveform, a, axis=1)
        p = p_new
        if i+1 in data:
            # print("L = ", i+1)
            data_tindex.append(len(waveform[0])-1)
            data_prec.append(copy.deepcopy(p))
            # print(data_tindex, data_prec)
    return waveform, data_tindex, data_prec

def time_evolve_2(H_seq, dt, rho_0):
    rho = rho_0
    rho_list = np.empty((len(H_seq), 4, 4), dtype=np.complex)
    U_total = np.identity(4)
    for i in range(len(H_seq)):
        H = H_seq[i]
        U = expm(-1j * H * dt)
        U_total = U @ U_total
        rho = U @ rho @ U.conj().T
        rho_list[i] = rho
    return rho_list, U_total

def inverse_gate_apply(rho_list, data_tindex, inverse_set, prec, dt):
    pre_measure_rho = np.empty((len(data_tindex), 4, 4), dtype=np.complex)
    for i in range(len(data_tindex)):
        U_inv = inverse_set[i]
        tindex = data_tindex[i]
        phase = prec[i]
        U = U_inv @ r(dt * tindex) @ np.diag(phase).conj()
        # print("tindex:", data_tindex[i])
        # print("U: ", U)
        # print("rho before measure: ", rho_list[tindex])
        rho = U @ rho_list[tindex] @ U.conj().T
        pre_measure_rho[i] = rho
    return pre_measure_rho

def waveform_2_H(waveform, dt, f):
    time_list = dt * np.array(range(1, len(waveform[0])+1))
    iq = np.array([[sum(np.exp(2 * np.pi * -1j * np.transpose(f) * time_list) * waveform)]])
    iq = np.moveaxis(iq, -1, 0)
    esr = np.array([[0, Omega, Omega, 0],
                    [0, 0, 0, Omega],
                    [0, 0, 0, Omega],
                    [0, 0, 0, 0]]) * iq
    esr = esr + np.transpose(esr.conj(), (0, 2, 1))
    H_seq = 1 / 2 * 2 * np.pi * esr + Hd
    # H_seq = 1 / 2 * 2 * np.pi * esr
    return H_seq



