import random
import matplotlib.pyplot as plt
import multiprocessing as mp
from scipy.optimize import curve_fit
from lib.nswrb import *

# stochastic noise deviation
std_uu = 16100
std_ud = 10100
std_du = 21000
std_dd = 0
four_noise = [std_uu, std_ud, std_du, std_dd]

offset_f = ac_stark_frequency()
phase_comp = ac_stark_modulation(offset_f, T_pi_2)
# phase_comp = np.ones((4, 4))
L = [1, 3, 5, 7, 10]
F = np.zeros(len(L))
dt = 5e-10
f = np.array([[f_1u, f_1d, f_2u, f_2d]])
rho_0 = error_initial_state(0, 0, 0)
rep = 3

def RB_single_sequence(l, noise, rho_initial, delta_t, phase_compensation, four_frequency):
    cliff_seq = random.choices(Cliff_decompose, k=l[-1])
    wav, tindex, p_rec = generate_cliff_waveform(cliff_seq, l, dt, phase_compensation)

    # add noise here
    sf1 = np.random.normal(0.0, noise[0])
    sf2 = np.random.normal(0.0, noise[1])
    sf3 = np.random.normal(0.0, noise[2])
    sf4 = np.random.normal(0.0, noise[3])
    H_noise = 2 * np.pi * np.diag([sf1, sf2, sf3, sf4])
    # end noise

    H_seq = waveform_2_H(wav, delta_t, four_frequency) + H_noise
    # H_seq = waveform_2_H(wav, delta_t, four_frequency)
    rho_list, U = time_evolve_2(H_seq, delta_t, rho_initial)
    inv = get_perfect_inverse_set(cliff_seq, l)
    rho_data = inverse_gate_apply(rho_list, tindex, inv, p_rec, delta_t)
    f = np.zeros(len(l))
    for i in range(len(f)):
        fidelity = abs(np.trace(rho_initial @ rho_data[i]))
        f[i] += fidelity
    return f


result_list = []
def log_result(result):
    result_list.append(result)

# Fitting function
def func(x, A, B, r):
    return A * (1 - 4 / 3 * r) ** x + B


if __name__ == '__main__':
    pool = mp.Pool()
    for re in range(rep):
        pool.apply_async(RB_single_sequence, args=(L, four_noise, rho_0, dt, phase_comp, f), callback=log_result)
    pool.close()
    pool.join()
    F = sum(result_list) / rep
    print(F)

    f5 = open('2q_lab_RB_simu_L.pkl', 'wb')
    pickle.dump(L, f5)
    f5.close()

    f6 = open('2q_lab_RB_simu_y.pkl', 'wb')
    pickle.dump(F, f6)
    f6.close()

    popt, pcov = curve_fit(func, L, F, p0=[1, 0, 0], bounds=(0, 1), maxfev=5000)
    # p0 is the guess of the parameters.
    # Guess B ~ 0 (ideally be 0.25) and r ~ 0 (no noise model now so r should be ultra low)
    print("F_Ciff = 1 - r = ", 1 - popt[2])
    print("A = ", popt[0])
    print("B = ", popt[1])

    plt.plot(L, F, 'o', markersize=4)
    plt.plot(L, func(L, *popt), 'r-')
    # plt.plot(x, func(x, 0.75, 0.25, 0.053), 'b-')
    plt.ylim(top=1.0)
    plt.xlabel("Number of Cliffords (L)")
    plt.ylabel("Proj. State Prob.")
    plt.title("Two-qubit RB Fitting")
    plt.show()
