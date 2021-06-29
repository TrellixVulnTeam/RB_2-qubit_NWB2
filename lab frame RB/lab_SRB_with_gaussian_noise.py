import random
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from lib.nswrb import *

# stochastic noise deviation
std_uu = 20000
std_ud = 20000
std_du = 20000
std_dd = 0


offset_f = ac_stark_frequency()
phase_comp = ac_stark_modulation(offset_f, T_pi_2)
# phase_comp = np.ones((4, 4))
L = [1, 3, 5, 7, 10, 20, 30]
F = np.zeros(len(L))
dt = 5e-10
f = np.array([[f_1u, f_1d, f_2u, f_2d]])
rho_0 = error_initial_state(0, 0, 0)
rep = 1


for re in range(rep):
    cliff_seq = random.choices(Cliff_decompose, k=L[-1])
    # cliff_seq = [[10, 3, 0, 9], [10, 3, 6, 13, 4, 9, 11]]
    # cliff_seq = [[7, 4, 9, 1, 9, 11], [12, 0, 10, 5, 11, 2, 8, 12], [13, 4, 10, 1, 10, 12], [0, 9, 3, 8, 13]]
    # cliff_seq = [[0, 9, 3, 8, 13, 13, 4, 10, 1, 10, 12, 12, 0, 10, 5, 11, 2, 8, 12, 7, 4, 9, 1, 9, 11]]
    print(cliff_seq)
    wav, tindex, p_rec = generate_cliff_waveform(cliff_seq, L, dt, phase_comp)
    # print("tindex: ", tindex)
    # print("phase_rec: ", p_rec)
    H_seq = waveform_2_H(wav, dt, f)
    rho_list, U = time_evolve_2(H_seq, dt, rho_0)
    inv = get_perfect_inverse_set(cliff_seq, L)
    C = get_perfect_cliff([0])
    rho_data = inverse_gate_apply(rho_list, tindex, inv, p_rec, dt)
    for i in range(len(F)):
        fidelity = abs(np.trace(rho_0 @ rho_data[i]))
        print(fidelity)
        F[i] += fidelity/rep


# Fitting function
def func(x, A, B, r):
    return A * (1 - 4 / 3 * r) ** x + B

print(F)
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
