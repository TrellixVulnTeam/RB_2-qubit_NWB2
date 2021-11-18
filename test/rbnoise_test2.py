from lib.rbnoise import *
import matplotlib.pyplot as plt

'''''''''
matching correlation function of OU noise with theory
'''''''''

dt = 5e-8
N = 1500
gamma = 1e4
Gamma = gamma/(2 * np.pi)
sigma = 1

X = np.arange(0, (N-1/2)*dt, dt)
X_spec = np.fft.rfftfreq(N, dt)

plt.figure(1)
plt.plot(X[1:100], ou_noise_seq(N, dt, gamma, sigma)[1:100])
plt.show()

averaged_correlation_func = np.zeros(N)
shot = 500
for j in range(shot):
    noise_seq = ou_noise_seq(N, dt, gamma, sigma)
    c = correlation_function(noise_seq)
    # print(c)
    averaged_correlation_func = averaged_correlation_func + c
averaged_correlation_func = averaged_correlation_func/shot

C_thr = sigma**2 * np.exp(-gamma*X)
print("ok!!")
plt.figure(2)

S = corr_func_to_psd(averaged_correlation_func, X_spec, dt)
S_plt = np.abs(S)
S_thr = psd_ou(X_spec, sigma, gamma)
S_corr = corr_func_to_psd(C_thr, X_spec, dt)
S_corr_plt = np.abs(S_corr)
plt.plot(X_spec, S_plt)
plt.plot(X_spec, S_thr)
plt.plot(X_spec, S_corr_plt)
plt.legend(['S from noise shots', 'S theoretical', 'S from theoretical C'])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.yscale("log")
plt.xscale("log")
plt.show()

plt.figure(3)
plt.plot(X[1:100], averaged_correlation_func[1:100], 'b-')
plt.plot(X[1:100], C_thr[1:100], 'r-')
plt.title("OU correlation function test")
plt.legend(['C(t) from noise shots', 'C(t) from theory'])
plt.xlabel('t2-t1 (s)')
plt.ylabel('C(t1, t2)')
plt.show()
