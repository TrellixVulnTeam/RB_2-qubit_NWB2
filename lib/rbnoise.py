"""
RandomizedBenchmarkingNoise
=====

Provides
  1. Noise generating

"""
import numpy as np
import pickle

# Calculate correlation function of given discrete noise sequence
# Output would be a numpy array c with the same number of elements as input sequence
# c[0] is the correlation function for t1-t2 = 0; c[1] for t1-t2 = 1*dt etc.
def correlation_function(noise_seq):
    n = len(noise_seq)
    c = np.zeros(n)
    for i in range(n):
        for j in range(n-i):
            c[i] += noise_seq[j] * noise_seq[j+i]
        c[i] = c[i]/(n-i)
    return c

# Transform correlation function c[i] = c(i*dt) to PSD S(f)
def corr_func_to_psd(c, f, dt):
    integrand = dt * (c[1:] + c[:-1])/2
    omega = 2*np.pi*f
    omega = omega[..., None]    # turn omega into column vector
    n = len(integrand)
    tau = (np.arange(n) + 1/2)*dt
    fft_mat = 2 * integrand * np.cos(omega * tau)
    fft_one = np.ones(n)
    return fft_mat @ fft_one


'''''''''
OU noise
'''''''''
# OU noise: dx(t) = -gamma*x(t)*dt + sigma*sqrt(2*gamma)*dW(t) ; dW(t) ~ sqrt(dt)N(0, 1)
def ou_noise_seq(length, dt, gamma, sigma):
    y = np.zeros(length)
    np.random.seed()
    y[0] = np.random.normal(loc=0.0, scale=sigma)   # initial condition
    noise = np.random.normal(loc=0.0, scale=1, size=length) * np.sqrt(dt)    # define noise process
    # solve SDE
    for i in range(1, length):
        y[i] = (1 - gamma * dt) * y[i-1] + sigma * np.sqrt(2*gamma) * noise[i]
    return y

# return noise spectrum of OU noise from given sigma and Gamma = gamma/(2pi)
def psd_ou(f, sigma, gamma):
    b_gamma = gamma/(2 * np.pi)
    return sigma**2 * b_gamma / np.pi / (b_gamma**2 + f**2)


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

def find_sigma_ou(s_0, kappa, gamma_j_arr):
    b_gamma_j_arr = gamma_j_arr/(2 * np.pi)
    denominator_list = kappa**(np.log10(b_gamma_j_arr))/b_gamma_j_arr
    denominator = np.sum(denominator_list)
    return np.sqrt(np.pi * s_0 / denominator)

# generate one over f noise sequence from combining OU noise sequences
# range: tuple (a, b) s.t. one over f spectrum located between frequency (10^a, 10^b)
def one_over_f_noise_seq(length, dt, s_0, alpha, noise_range=(7, 9), dev=0.5):
    gamma_arr = np.arange(noise_range[0], noise_range[1]+dev, dev)
    gamma_arr = 10**gamma_arr
    kappa = find_kappa(alpha)
    sigma_ou = find_sigma_ou(s_0, kappa, gamma_arr)
    sigma_arr = sigma_ou * kappa**(np.log10(gamma_arr)/2)
    # broadcasting gamma_arr and sigma_arr
    ou_seqs = np.array([ou_noise_seq(length, dt, gamma_arr[i], sigma_arr[i]) for i in range(len(gamma_arr))])
    return np.sum(ou_seqs, axis=0)

# return noise spectrum of 1/f noise from given kappa, sigma and lists of Gamma = gamma/(2pi) of combined OU noise
def psd_one_over_f(f, kappa, gamma_arr, sigma):
    b_gamma_arr = gamma_arr/(2 * np.pi)
    sigma_arr = sigma * kappa**(np.log10(b_gamma_arr)/2)
    psd_arr = np.array([psd_ou(f, sigma_arr[i],  gamma_arr[i]) for i in range(len(sigma_arr))])
    return sum(psd_arr)
