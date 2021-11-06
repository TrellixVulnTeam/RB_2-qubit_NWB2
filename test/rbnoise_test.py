from lib.rbnoise import *
from scipy.fftpack import fft, ifft, fftfreq
import matplotlib.pyplot as plt

dt = 5e-9
N = 1000
s_0 = 1
alpha = [0.5, 1, 1.5, 2]
noise_rng = (-7, 5)
dev = 0.5
one_over_f_noise_seq(N, dt, s_0, alpha[1], noise_range=noise_rng)
S_plt_list = []

X_spec = fftfreq(N, dt)[:N//2]

# for i in range(len(alpha)):
#     averaged_correlation_func = np.zeros(N)
#     for j in range(100):
#         noise_seq = one_over_f_noise_seq(N, dt, s_0, alpha[i], noise_range=noise_rng, dev=dev)
#         c = correlation_function(noise_seq)
#         averaged_correlation_func = averaged_correlation_func + c
#     averaged_correlation_func = [x/100 for x in averaged_correlation_func]
#     S = fft(averaged_correlation_func)
#     S_plt_list.append(2.0/N * np.abs(S[0:N//2]))
#     plt.plot(X_spec, S_plt_list[i])

averaged_correlation_func = np.zeros(N)

for j in range(100):
    noise_seq = one_over_f_noise_seq(N, dt, s_0, alpha[1], noise_range=noise_rng, dev=dev)
    c = correlation_function(noise_seq)
    averaged_correlation_func = averaged_correlation_func + c

S_ = fft(averaged_correlation_func)
S = 2.0/N * np.abs(S_[0:N//2])
plt.plot(X_spec, S)

# plt.plot(X_spec, S_plt1)

gamma_list = np.arange(noise_rng[0], noise_rng[1]+dev, dev)
gamma_list = [10**x for x in gamma_list]
Gamma_list = [x/(2*np.pi) for x in gamma_list]
kappa = find_kappa(1)
S_thr = [S_one_over_f(x, kappa, Gamma_list, find_sigma_OU(1, kappa, Gamma_list)) for x in X_spec]
plt.plot(X_spec, S_thr)

index = int((np.where(X_spec == X_spec[np.abs(X_spec - 10**6).argmin()])[0]))
print(X_spec[index])
print(S[index])
print(S_thr[index])

plt.yscale("log")
plt.xscale("log")
plt.show()

