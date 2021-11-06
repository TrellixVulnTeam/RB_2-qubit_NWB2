from scipy.fftpack import fft, ifft, fftfreq
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

Gamma = 0.5
Sigma = 1
# OU noise: dx(t) = -gamma*x(t)*dt + sigma*sqrt(2*gamma)*dW(t) ; dW(t) ~ sqrt(dt)N(0, 1)
def OU_noise_seq(length, dt, gamma, sigma):
    y = np.zeros(length)
    # y[0] = np.random.normal(loc=0.0, scale=std)  # initial condition
    y[0] = 0
    noise = np.random.normal(loc=0.0, scale=1, size=length) * np.sqrt(dt)    # define noise process
    # solve SDE
    for i in range(1, length):
        y[i] = (1 - gamma * dt) * y[i-1] + sigma * np.sqrt(2*gamma) * noise[i]
    return y


# generate one over f noise sequence
# range: tuple (a, b) s.t. one over f spectrum located between frequency (10^a, 10^b)
def one_over_f_noise_seq(length, dt, sigma_OU, alpha=1, noise_range=(7, 9), dev=0.5):
    gamma = np.arange(noise_range[0], noise_range[1]+dev, dev)
    gamma = [10**x for x in gamma]
    Gamma = [x/(2*np.pi) for x in gamma]
    kappa = 0.316   # TODO: kappa-alpha relation TBD
    sigma = [sigma_OU * kappa**(np.log10(x)/2) for x in Gamma]
    OU_seqs = [OU_noise_seq(length, dt, gamma[i], sigma[i]) for i in range(len(gamma))]
    return np.array([sum(x) for x in zip(*OU_seqs)])

def diagonal_OU_noise(waveform, dt, gamma, sigma):
    N = len(waveform[0])
    sf1 = OU_noise_seq(N, dt, gamma, sigma)
    sf2 = OU_noise_seq(N, dt, gamma, sigma)
    sf3 = OU_noise_seq(N, dt, gamma, sigma)
    sf4 = OU_noise_seq(N, dt, gamma, sigma)
    noisy_h = np.empty((N, 4, 4))
    for i in range(N):
        noisy_h[i] = np.diag([sf1[i], sf2[i], sf3[i], sf4[i]])
    return noisy_h

def correlation_function(noise_seq):
    N = len(noise_seq)
    c = np.zeros(N)
    for i in range(N):
        for j in range(N-i):
            c[i] += noise_seq[j] * noise_seq[j+i]
        c[i] = c[i]/(N-i)
    return c

def s_j(f, sigma_j, Gamma_j):
    return sigma_j**2 * Gamma_j / np.pi / (Gamma_j**2 + f**2)

def S(f, kappa, Gamma_j_list, sigma):
    sigma_j_list = [sigma * kappa**(np.log10(x)/2) for x in Gamma_j_list]
    s_j_list = [s_j(f, sigma_j_list[i], Gamma_j_list[i]) for i in range(len(Gamma_j_list))]
    return sum(s_j_list)

def find_sigma_OU(s_0, kappa, Gamma_j_list):
    deno_list = [kappa**(np.log10(x)/2)/x for x in Gamma_j_list]
    deno = sum(deno_list)
    return np.pi * s_0 / deno

def func(f, a, A):
    return A * 1/(f**a)

def func2(x, logA, alpha):
    return logA - alpha * x


dt = 5e-9
N = 500

X = [i*dt for i in range(N)]
# Y0 = OU_noise_seq(N, dt, Gamma, Sigma)
# Y1 = OU_noise_seq(N, dt, Gamma, Sigma)
# Y2 = OU_noise_seq(N, dt, Gamma, Sigma)
# Y3 = OU_noise_seq(N, dt, Gamma, Sigma)
Z0 = one_over_f_noise_seq(N, dt, Sigma, noise_range=(4, 7))
# Z1 = one_over_f_noise_seq(N, dt, Sigma, noise_range=(2, 7))

# plot1 = plt.figure(1)
# plt.plot(X, Y0)
# plt.plot(X, Y1)
# plt.plot(X, Y2)
# plt.plot(X, Y3)
# Z0_plot = [x/5 for x in Z0]
# Z1_plot = [x/5 for x in Z1]
# plt.plot(X, Z0_plot)
# plt.show()

averaged_correlation_func = np.zeros(N)
for i in range(100):
    noise_seq = one_over_f_noise_seq(N, dt, Sigma, noise_range=(4, 7))
    c = correlation_function(noise_seq)
    averaged_correlation_func = averaged_correlation_func + c
averaged_correlation_func = [x/100 for x in averaged_correlation_func]

plot2 = plt.figure(2)
# Z0_C = correlation_function(Z0_plot)
plt.plot(X, averaged_correlation_func)
plt.show()

plot3 = plt.figure(3)
Z0_S = fft(averaged_correlation_func)
Z0_fft_plt = 2.0/N * np.abs(Z0_S[0:N//2])
X_spec = fftfreq(N, dt)[:N//2]
log_X_spec = np.log10(X_spec)
# Z1_fft_plt = 2.0/N * np.abs(Z1_fft[0:N//2])

noise_range = (-7, 5)
dev = 0.5
gamma_list = np.arange(noise_range[0], noise_range[1]+dev, dev)
gamma_list = [10**x for x in gamma_list]
Gamma_list = [x/(2*np.pi) for x in gamma_list]
S_alpha = [S(x, 0.316, Gamma_list, 1) for x in X_spec]

plt.plot(X_spec, Z0_fft_plt)
# plt.plot(X_spec, Z1_fft_plt)
plt.plot(X_spec, func(X_spec, 1.5, Z0_fft_plt[25] * X_spec[25]))
plt.plot(X_spec, S_alpha)
plt.yscale("log")
plt.xscale("log")
plt.show()

# wav = np.zeros((4, 100000))
# noisy_H = diagonal_OU_noise(wav, dt, Gamma, Sigma)
# print(wav)
# print(noisy_H)
