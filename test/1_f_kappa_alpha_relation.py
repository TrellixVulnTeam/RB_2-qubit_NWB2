import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import exp10
from lib.rbnoise import *

def func(x, A, B):
    return A * 1/x + B


alpha_sample = [0.5, 0.8, 0.9, 1, 1.01, 1.1, 1.5, 2]
kappa_sample = [3.162, 1.585, 1.259, 1, 0.977, 0.795, 0.316, 0.01]

# def func(x):
#     return 2.12444824 * 1/x + -1.09983209

def func2(x, logA, alpha):
    return logA - alpha * x


noise_range = (-7, 5)
fitting_range = (0, 1)
plot_range = (-10, 10)
dev = 0.5
gamma_list = np.arange(noise_range[0], noise_range[1]+dev, dev)
gamma_list = [10**x for x in gamma_list]
Gamma_list = [x/(2*np.pi) for x in gamma_list]
kappa_list = [0.01*(x+1) for x in range(2000)]
alpha_list = np.zeros(len(kappa_list))

N = 1000
log_f = np.array([plot_range[0]+i*(plot_range[1] - plot_range[0])/(N-1) for i in range(N)])
f = [10**x for x in log_f]

index_left = int((np.where(log_f == log_f[np.abs(log_f - fitting_range[0]).argmin()])[0]))
index_right = int((np.where(log_f == log_f[np.abs(log_f - fitting_range[1]).argmin()])[0])+1)
log_f_fitting = log_f[index_left:index_right]
f_fitting = f[index_left:index_right]

for i in range(len(kappa_list)):
    S_kappa_fitting = [S_one_over_f(x, kappa_list[i], Gamma_list, 1) for x in f_fitting]
    popt, pcov = curve_fit(func2, log_f_fitting, np.log10(S_kappa_fitting), maxfev=5000)
    alpha_list[i] = popt[1]

# plt.plot(alpha_sample, kappa_sample, 'bo', markersize=3, label='samples', zorder=10)
plt.plot(alpha_list, kappa_list, 'go', markersize=2, label='my simulation', zorder=5)
x_plt = np.array([0.5+i*1.5/(N-1) for i in range(N)])
# plt.plot(x_plt, func(x_plt), 'r-', label='samples fitting')
plt.xlabel("alpha")
plt.ylabel("kappa")
plt.legend()
plt.show()

''''''''''''''''''''
S_alpha_1 = [S_one_over_f(x, find_kappa(0.5), Gamma_list, find_sigma_OU(1, find_kappa(0.5), Gamma_list)) for x in f]
S_alpha_2 = [S_one_over_f(x, find_kappa(1), Gamma_list, find_sigma_OU(1, find_kappa(1), Gamma_list)) for x in f]
S_alpha_3 = [S_one_over_f(x, find_kappa(1.5), Gamma_list, find_sigma_OU(1, find_kappa(1.5), Gamma_list)) for x in f]
S_alpha_4 = [S_one_over_f(x, find_kappa(2), Gamma_list, find_sigma_OU(1, find_kappa(2), Gamma_list)) for x in f]

S_alpha_1_fitting = S_alpha_1[index_left:index_right]
S_alpha_2_fitting = S_alpha_2[index_left:index_right]
S_alpha_3_fitting = S_alpha_3[index_left:index_right]
S_alpha_4_fitting = S_alpha_4[index_left:index_right]
popt1, pcov1 = curve_fit(func2, log_f_fitting, np.log10(S_alpha_1_fitting), maxfev=5000)
popt2, pcov2 = curve_fit(func2, log_f_fitting, np.log10(S_alpha_2_fitting), maxfev=5000)
popt3, pcov3 = curve_fit(func2, log_f_fitting, np.log10(S_alpha_3_fitting), maxfev=5000)
popt4, pcov4 = curve_fit(func2, log_f_fitting, np.log10(S_alpha_4_fitting), maxfev=5000)
plt.plot(log_f, S_alpha_1)
plt.plot(log_f, S_alpha_2)
plt.plot(log_f, S_alpha_3)
plt.plot(log_f, S_alpha_4)
plt.plot(log_f_fitting, exp10(func2(log_f_fitting, *popt1)), 'r-', label='kappa=3.2 (alpha=0.5 from fitting)')
plt.plot(log_f_fitting, exp10(func2(log_f_fitting, *popt2)), 'k-', label='kappa=1.0 (alpha=1.0 from fitting)')
plt.plot(log_f_fitting, exp10(func2(log_f_fitting, *popt3)), 'b-', label='kappa=0.32 (alpha=1.5 from fitting)')
plt.plot(log_f_fitting, exp10(func2(log_f_fitting, *popt4)), 'r-', label='kappa=0.01 (alpha=2 from fitting)')

index = int((np.where(log_f == log_f[np.abs(log_f - 6).argmin()])[0]))
print(log_f[index])
print(S_alpha_2[index])
plt.yscale("log")
plt.xlabel("log f")
plt.ylabel("S(f)")
plt.legend()
plt.show()

''''''''''''''''''''
