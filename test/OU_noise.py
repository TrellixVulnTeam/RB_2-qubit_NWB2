
"""
brownian() implements one dimensional Brownian motion (i.e. the Wiener process).
"""

from math import sqrt
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

Theta = 0
Sigma = 1
# OU noise: dx(t) = theta(u-x(t))dt + sigma*dW(t) ; dW(t) ~ sqrt(dt)N(0, std)
def OU_noise_seq(length, dt, theta, sigma, std, mu=0):
    y = np.zeros(length)
    # y[0] = np.random.normal(loc=0.0, scale=std)  # initial condition
    y[0] = 0
    noise = np.random.normal(loc=0.0, scale=std, size=length) * np.sqrt(dt)    # define noise process
    # solve SDE
    for i in range(1, length):
        y[i] = y[i-1] + theta * (mu - y[i-1])*dt + sigma*noise[i]
    return y

# def diagonal_OU_noise(waveform, dt, theta, sigma, u=0):
#     length = len(waveform[0])
#     t = np.linspace(t_0, t_end, length)  # define time axis
#     y = np.zeros(length)

def diagonal_OU_noise(waveform, dt, theta, sigma, std, u=0):
    N = len(waveform[0])
    sf1 = OU_noise_seq(N, dt, theta, sigma, std[0], mu=u)
    sf2 = OU_noise_seq(N, dt, theta, sigma, std[1], mu=u)
    sf3 = OU_noise_seq(N, dt, theta, sigma, std[2], mu=u)
    sf4 = OU_noise_seq(N, dt, theta, sigma, std[3], mu=u)
    noisy_h = np.empty((N, 4, 4))
    for i in range(N):
        noisy_h[i] = np.diag([sf1[i], sf2[i], sf3[i], sf4[i]])
    return noisy_h


dt = 5e-9
N = 500000
std_uu = 16100
std_ud = 10100
std_du = 21000
std_dd = 0
four_noise = [std_uu, std_ud, std_du, std_dd]
#
# X = [i*dt for i in range(N)]
# Y0 = OU_noise_seq(N, dt, Theta, Sigma, four_noise[0])
# Y1 = OU_noise_seq(N, dt, Theta, Sigma, four_noise[1])
# Y2 = OU_noise_seq(N, dt, Theta, Sigma, four_noise[2])
# Y3 = OU_noise_seq(N, dt, Theta, Sigma, four_noise[3])
# plt.plot(X, Y0)
# # plt.plot(X, Y1)
# # plt.plot(X, Y2)
# # plt.plot(X, Y3)
# plt.show()

wav = np.zeros((4, 100000))
noisy_h = diagonal_OU_noise(wav, dt, Theta, Sigma, four_noise)
print(wav)
print(noisy_h)
