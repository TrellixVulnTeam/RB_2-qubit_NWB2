import random
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from lib.nswrb import *

# Fitting function
def func(x, A, B, r):
    return A * (1 - 4 / 3 * r) ** x + B


# stochastic noise deviation
std_uu = 16100
std_ud = 10100
std_du = 21000
std_dd = 0

offset_f = ac_stark_frequency()
phase_comp = ac_stark_modulation(offset_f, T_pi_2)
# phase_comp = np.ones((4, 4))

# define the gate length series chosen to run RB protocol
L = [1, 3, 5]
# l1 = np.arange(1, 20, 2)
# l2 = np.arange(20, 40, 5)
# l3 = np.arange(40, 240, 20)
# l4 = np.arange(240, 300, 40)
# l5 = np.arange(300, 550, 50)
# L = np.hstack((l1, l2, l3, l4, l5))

# define Gaussian error standard deviation series
# sigma_list = 2000 * np.array(range(26))
sigma_list = [0, 2000]

dt = 5e-9
f = np.array([[f_1u, f_1d, f_2u, f_2d]])
rho_0 = error_initial_state(0, 0, 0)
rep = 2
F_Clifford = np.zeros(len(sigma_list))

for i in range(len(sigma_list)):
    F = np.zeros(len(L))
    for re in range(rep):
        cliff_seq = random.choices(Cliff_decompose, k=L[-1])
        wav, tindex, p_rec = generate_cliff_waveform(cliff_seq, L, dt, phase_comp)

        # add noise here
        sf1 = np.random.normal(0.0, sigma_list[i])
        sf2 = np.random.normal(0.0, sigma_list[i])
        sf3 = np.random.normal(0.0, sigma_list[i])
        H_noise = 2 * np.pi * np.diag([sf1, sf2, sf3, 0])
        # end noise

        H_seq = waveform_2_H(wav, dt, f) + H_noise
        # H_seq = waveform_2_H(wav, dt, f)
        rho_list, U = time_evolve_2(H_seq, dt, rho_0)
        inv = get_perfect_inverse_set(cliff_seq, L)
        C = get_perfect_cliff([0])
        rho_data = inverse_gate_apply(rho_list, tindex, inv, p_rec, dt)
        for j in range(len(F)):
            fidelity = abs(np.trace(rho_0 @ rho_data[j]))
            print(fidelity)
            F[j] += fidelity/rep

    popt, pcov = curve_fit(func, L, F, p0=[1, 0, 0], bounds=(0, 1), maxfev=5000)
    F_Clifford[i] = popt[2] * 100

f5 = open('quasi_noise_sigma.pkl', 'wb')
pickle.dump(sigma_list, f5)
f5.close()

f6 = open('quasi_noise_infidelity.pkl', 'wb')
pickle.dump(F_Clifford, f6)
f6.close()

plt.plot(sigma_list, F_Clifford, 'o', markersize=4)
plt.xlabel("Frequency noise (kHz)")
plt.ylabel("Clifford infidelity (%)")
plt.show()
