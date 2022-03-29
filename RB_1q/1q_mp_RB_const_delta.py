import random
import matplotlib.pyplot as plt
import multiprocessing as mp
import copy
from scipy.optimize import curve_fit
from lib.oneqrb import *


def RB_single_sequence(l, delta, rho_initial, delta_t, noise_type):
    np.random.seed()
    cliff_seq = np.random.choice(24, l[-1], replace=True)
    # add noise here
    noise_seq = delta * np.ones(len(cliff_seq))
    # end of adding noise
    seq = get_seq_1q(cliff_seq, noise_seq, delta_t=delta_t, noise_type=noise_type)
    f = np.zeros(len(l))
    rho = copy.deepcopy(rho_initial)
    for i in range(len(seq)):
        rho = seq[i] @ rho @ seq[i].conj().T
        if i+1 in l:
            inv = get_seq_inverse(cliff_seq[:(i+1)])
            rho_inversed = inv @ rho @ inv.conj().T
            fidelity = abs(np.trace(rho_initial @ rho_inversed))
            j = l.index(i+1)
            f[j] += fidelity
    return f

# Fitting function
def func(x, B, r):
    return 1/2 * (1 - 2 * r) ** x + B


L = [1, 3, 5, 7, 10]
dt = 100
noise_type = HAMILTONIAN_NOISE
rho_0 = np.array([[1, 0],
                  [0, 0]])
rep = 3


# constant noise angle TODO: noise frequency here
delta_list = [x * 0.01 for x in list(range(1, 51))]
# delta_list = [0.05, 0.1, 0.15, 0.2]

F_Clifford = np.zeros(len(delta_list))
r_sqrd = np.zeros(len(delta_list))


if __name__ == '__main__':
    for i in range(len(delta_list)):
        result_list = []

        def log_result(result):
            result_list.append(result)

        pool = mp.Pool()
        for re in range(rep):
            pool.apply_async(RB_single_sequence, args=(L, delta_list[i], rho_0, dt, noise_type), callback=log_result)
        pool.close()
        pool.join()
        F = sum(result_list) / rep
        print(F)

        ff = open(str(delta_list[i]) + "_1q.pkl", "wb")
        pickle.dump((delta_list[i], F), ff)
        ff.close()

        popt, pcov = curve_fit(func, L, F, p0=[1, 0, 0], bounds=(0, 1), maxfev=5000)
        F_Clifford[i] = (1 - popt[1]) * 100

        residuals = F - func(L, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((F - np.mean(F))**2)
        r_sqrd[i] = 1 - (ss_res/ss_tot)

    print(F_Clifford)
    print(r_sqrd)

    f5 = open('const_delta_list_1q.pkl', 'wb')
    pickle.dump(delta_list, f5)
    f5.close()

    f6 = open('const_delta_fidelity_1q.pkl', 'wb')
    pickle.dump(F_Clifford, f6)
    f6.close()

    f7 = open('const_delta_list_r_squared_1q.pkl', 'wb')
    pickle.dump(r_sqrd, f7)
    f7.close()

    plot1 = plt.figure(1)
    plt.plot(delta_list, F_Clifford, 'o', markersize=4)
    plt.xlabel("Dephasing noise angle (rad)")
    plt.ylabel("Clifford fidelity (%)")
    plt.show()

    plot2 = plt.figure(2)
    plt.plot(delta_list, r_sqrd, 'o', markersize=4)
    plt.xlabel("Dephasing noise angle (rad)")
    plt.ylabel("R_squared")
    plt.show()

