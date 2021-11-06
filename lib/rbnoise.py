"""
RandomizedBenchmarkingNoise
=====

Provides
  1. Noise generating

"""
import numpy as np
import pickle


'''''''''
OU noise
'''''''''

# OU noise: dx(t) = -gamma*x(t)*dt + sigma*sqrt(2*gamma)*dW(t) ; dW(t) ~ sqrt(dt)N(0, 1)
def OU_noise_seq(length, dt, gamma, sigma):
    y = np.zeros(length)
    y[0] = np.random.normal(loc=0.0, scale=sigma**2)   # initial condition
    noise = np.random.normal(loc=0.0, scale=1, size=length) * np.sqrt(dt)    # define noise process
    # solve SDE
    for i in range(1, length):
        y[i] = (1 - gamma * dt) * y[i-1] + sigma * np.sqrt(2*gamma) * noise[i]
    return y

# return noise spectrum of OU noise from given sigma and Gamma = gamma/(2pi)
def S_OU(f, sigma, Gamma):
    return sigma**2 * Gamma / np.pi / (Gamma**2 + f**2)


'''''''''
1/f^(alpha) noise
'''''''''
# alpha for 1/f^(alpha) noise
with open("../lib/one_over_f_alpha_list.pkl", "rb") as f1:
    one_over_f_alpha = pickle.load(f1)
f1.close()

# kappa for 1/f^(alpha) noise with different alpha
with open("../lib/one_over_f_kappa_list.pkl", "rb") as f2:
    one_over_f_kappa = pickle.load(f2)
f2.close()

# find kappa for generating 1;f^(alpha) noise with given alpha
def find_kappa(alpha):
    index = int((np.where(one_over_f_alpha == one_over_f_alpha[np.abs(one_over_f_alpha - alpha).argmin()])[0]))
    return one_over_f_kappa[index]

def find_sigma_OU(s_0, kappa, Gamma_j_list):
    deno_list = [kappa**(np.log10(x))/x for x in Gamma_j_list]
    deno = sum(deno_list)
    return np.sqrt(np.pi * s_0 / deno)

# generate one over f noise sequence from combining OU noise sequences
# range: tuple (a, b) s.t. one over f spectrum located between frequency (10^a, 10^b)
def one_over_f_noise_seq(length, dt, s_0, alpha, noise_range=(7, 9), dev=0.5):
    gamma_list = np.arange(noise_range[0], noise_range[1]+dev, dev)
    gamma_list = [10**x for x in gamma_list]
    Gamma_list = [x/(2*np.pi) for x in gamma_list]
    kappa = find_kappa(alpha)
    print("kappa=", kappa)
    sigma_OU = find_sigma_OU(s_0, kappa, Gamma_list)
    sigma = [sigma_OU * kappa**(np.log10(x)/2) for x in Gamma_list]
    OU_seqs = [OU_noise_seq(length, dt, gamma_list[i], sigma[i]) for i in range(len(gamma_list))]
    return np.array([sum(x) for x in zip(*OU_seqs)])

# Calculate correlation function of given discrete noise sequence
# Output would be a numpy array c with the same number of elements as input sequence
# c[0] is the correlation function for t1-t2 = 0; c[1] for t1-t2 = 1*dt etc.
def correlation_function(noise_seq):
    N = len(noise_seq)
    c = np.zeros(N)
    for i in range(N):
        for j in range(N-i):
            c[i] += noise_seq[j] * noise_seq[j+i]
        c[i] = c[i]/(N-i)
    return c

# return noise spectrum of 1/f noise from given kappa, sigma and lists of Gamma = gamma/(2pi) of combined OU noise
def S_one_over_f(f, kappa, Gamma_list, sigma):
    sigma_list = [sigma * kappa**(np.log10(x)/2) for x in Gamma_list]
    S_list = [S_OU(f, sigma_list[i], Gamma_list[i]) for i in range(len(Gamma_list))]
    return sum(S_list)
