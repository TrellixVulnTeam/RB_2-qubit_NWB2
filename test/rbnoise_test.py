from lib.rbnoise import *
import matplotlib.pyplot as plt

dt = 5e-9
N = 1000
s_0 = 1
alpha = [0.5, 1, 1.5, 2]
noise_rng = (-7, 7)
plot_range = (-10, 10)
dev = 0.5

step_f = 0.001
log_f = np.arange(plot_range[0], plot_range[1] + step_f, step_f)
f = 10**log_f

gamma_list = np.arange(noise_rng[0], noise_rng[1]+dev, dev)
gamma_list = 10**gamma_list

shots = 100
for i in range(len(alpha)):
    print('alpha = ' + str(alpha[i]) + ' is working...')
    averaged_correlation_func = np.zeros(N)
    for j in range(100):
        noise_seq = one_over_f_noise_seq(N, dt, s_0, alpha[i], noise_range=noise_rng, dev=dev)
        c = correlation_function(noise_seq)
        averaged_correlation_func = averaged_correlation_func + c

    averaged_correlation_func = averaged_correlation_func / shots
    S = corr_func_to_psd(averaged_correlation_func, f, dt)
    plt.plot(f, S, marker='o', markersize=3, label='S(alpha=' + str(alpha[i]) + ')')

    kappa = find_kappa(alpha[i])
    S_thr = psd_one_over_f(f, kappa, gamma_list, find_sigma_ou(1, kappa, gamma_list))
    plt.plot(f, S_thr)
    plt.plot(f, S, label='S_thr(alpha=' + str(alpha[i]) + ')')
    print('alpha = ' + str(alpha[i]) + ' is done!')

plt.legend()
plt.yscale("log")
plt.xscale("log")
plt.show()
