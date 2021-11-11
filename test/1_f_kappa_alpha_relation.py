import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import exp10
from lib.rbnoise import *

def func(x, a, b):
    return a * 1/x + b


# alpha_sample = [0.5, 0.8, 0.9, 1, 1.01, 1.1, 1.5, 2]
# kappa_sample = [3.162, 1.585, 1.259, 1, 0.977, 0.795, 0.316, 0.01]

# def func(x):
#     return 2.12444824 * 1/x + -1.09983209

def func2(x, log_a, alpha):
    return log_a - alpha * x


noise_range = (-7, 7)
fitting_range = (-6, 2.5)
plot_range = (-10, 10)
dev = 0.5
step_kappa = 0.01
stop_kappa = 20
gamma_arr = np.arange(noise_range[0], noise_range[1]+dev, dev)
gamma_arr = 10**gamma_arr
kappa_arr = np.arange(step_kappa, stop_kappa + step_kappa, step_kappa)
alpha_arr = np.zeros(len(kappa_arr))

step_f = 0.001
log_f = np.arange(plot_range[0], plot_range[1] + step_f, step_f)
f = 10**log_f

# for i in range(len(kappa_arr)):
#     S_kappa_fitting = [psd_one_over_f(x, kappa_arr[i], gamma_arr, 1) for x in f_fitting]
#     popt, pcov = curve_fit(func2, log_f_fitting, np.log10(S_kappa_fitting), maxfev=5000)
#     alpha_arr[i] = popt[1]
#
# plt.plot(alpha_arr, kappa_arr, 'go', markersize=2, label='my simulation', zorder=5)
# plt.xlabel("alpha")
# plt.ylabel("kappa")
# plt.legend()
# plt.show()

''''''''''''''''''''
alpha_list = [0.5, 1.0, 1.5, 2.0]
index_left = int((np.where(log_f == log_f[np.abs(log_f - fitting_range[0]).argmin()])[0]))
index_right = int((np.where(log_f == log_f[np.abs(log_f - fitting_range[1]).argmin()])[0])+1)
log_f_fitting = log_f[index_left:index_right]
f_fitting = f[index_left:index_right]

for i in range(len(alpha_list)):
    S_alpha = psd_one_over_f(f, find_kappa(alpha_list[i]), gamma_arr,
                             find_sigma_ou(1, find_kappa(alpha_list[i]), gamma_arr))
    S_alpha_fitting = S_alpha[index_left:index_right]
    popt, pcov = curve_fit(func2, log_f_fitting, np.log10(S_alpha_fitting), maxfev=5000)
    plt.plot(log_f, S_alpha,
             label='alpha = ' + str(alpha_list[i]) + ' (alpha = ' + str(round(popt[1], 5)) + ' from fitting)')
    plt.plot(log_f_fitting, exp10(func2(log_f_fitting, *popt)))

plt.yscale("log")
plt.xlabel("log f")
plt.ylabel("Normalized S(f)")
plt.title('OU_composed 1/f^(alpha) noise')
plt.legend()
plt.show()
''''''''''''''''''''
