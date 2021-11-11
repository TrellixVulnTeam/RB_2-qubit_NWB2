import random
import matplotlib.pyplot as plt
import multiprocessing as mp
from scipy.optimize import curve_fit
from lib.nswrb import *

# TODO: gamma = (-3, 8)
# TODO: Find reasonable sigma_ou

def RB_single_sequence(l, noise_std, noise_switch, rho_initial, delta_t, phase_compensation, four_frequency):
    cliff_seq = random.choices(Cliff_decompose, k=l[-1])
    wav, tindex, p_rec = generate_cliff_waveform(cliff_seq, l, delta_t, phase_compensation)

    # add noise here
    # sf1 = np.random.normal(0.0, noise_std[0])
    # sf2 = np.random.normal(0.0, noise_std[1])
    # sf3 = np.random.normal(0.0, noise_std[2])
    # sf4 = np.random.normal(0.0, noise_std[3])
    # H_noise = 2 * np.pi * np.diag([sf1, sf2, sf3, sf4])
    # TODO: Change noise here
    H_noise = time_varying_gaussian_noise(wav, delta_t, noise_std, f_noise=noise_switch)
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

# Fitting function
def func(x, A, B, r):
    return A * (1 - 4 / 3 * r) ** x + B


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
dt = 5e-9
f = np.array([[f_1u, f_1d, f_2u, f_2d]])
rho_0 = error_initial_state(0, 0, 0)
rep = 3

# define noise changing frequencies #TODO: noise frequency here
# switch_list = 1000 * np.array(range(1, 31))
switch_list = [1000, 2000]

F_Clifford = np.zeros(len(switch_list))
r_sqrd = np.zeros(len(switch_list))


if __name__ == '__main__':
    for i in range(len(switch_list)):
        result_list = []

        def log_result(result):
            result_list.append(result)

        pool = mp.Pool()
        for re in range(rep):
            pool.apply_async(RB_single_sequence, args=(L, four_noise, switch_list[i],
                                                       rho_0, dt, phase_comp, f), callback=log_result)
        pool.close()
        pool.join()
        F = sum(result_list) / rep
        print(F)

        ff = open(str(switch_list[i]) + ".pkl", "wb")
        pickle.dump((switch_list[i], F), ff)
        ff.close()

        popt, pcov = curve_fit(func, L, F, p0=[1, 0, 0], bounds=(0, 1), maxfev=5000)
        F_Clifford[i] = popt[2] * 100

        residuals = F - func(L, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((F - np.mean(F))**2)
        r_sqrd[i] = 1 - (ss_res/ss_tot)

    print(F_Clifford)
    print(r_sqrd)

    f5 = open('noise_switch_frequency.pkl', 'wb')
    pickle.dump(switch_list, f5)
    f5.close()

    f6 = open('noise_switch_infidelity.pkl', 'wb')
    pickle.dump(F_Clifford, f6)
    f6.close()

    f7 = open('noise_switch_r_squared.pkl', 'wb')
    pickle.dump(r_sqrd, f7)
    f7.close()

    plot1 = plt.figure(1)
    plt_switch_list = [x / 1000 for x in switch_list]
    plt.plot(plt_switch_list, F_Clifford, 'o', markersize=4)
    plt.xlabel("Switching Frequency (kHz)")
    plt.ylabel("Clifford infidelity (%)")
    plt.show()

    plot2 = plt.figure(2)
    plt_switch_list = [x / 1000 for x in switch_list]
    plt.plot(plt_switch_list, r_sqrd, 'o', markersize=4)
    plt.xlabel("Switching Frequency (kHz)")
    plt.ylabel("R_squared")
    plt.show()

