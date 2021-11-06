from lib.rbnoise import *
from scipy.fftpack import fft, ifft, fftfreq
import matplotlib.pyplot as plt

'''''''''
test OU noise function
'''''''''

dt = 0.001
N = 2000
gamma = 1
Gamma = gamma/(2*np.pi)
sigma = 1
S_plt_list = []

X = [i*dt for i in range(N)]
X_spec = fftfreq(N, dt)[:N//2]

plt.figure(1)
plt.plot(X, OU_noise_seq(N, dt, gamma, sigma))
plt.show()

averaged_correlation_func = np.zeros(N)
shot = 500
for j in range(shot):
    noise_seq = OU_noise_seq(N, dt, gamma, sigma)
    c = correlation_function(noise_seq)
    # print(c)
    averaged_correlation_func = averaged_correlation_func + c
averaged_correlation_func = [x / shot for x in averaged_correlation_func]

# S = fft(averaged_correlation_func)
# S_plt = 2.0/N * np.abs(S[0:N//2])
# S_thr = [S_OU(x, sigma, Gamma) for x in X_spec]
print("ok!")
# plt.plot(X_spec, S_plt, 'b-')
# plt.plot(X_spec, S_thr, 'r-')
# plt.yscale("log")
# plt.xscale("log")
# plt.show()

C_thr = [(sigma**2 * np.exp(-2*np.pi*Gamma*x)) for x in X]
plt.figure(2)
plt.plot(X, averaged_correlation_func, 'b-')
plt.plot(X, C_thr, 'r-')
plt.title("OU correlation function test")
plt.show()
