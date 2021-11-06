from lib.rbnoise import *
from scipy.fftpack import fft, ifft, fftfreq
import matplotlib.pyplot as plt

N = 1000
dt = 0.002

seq = OU_noise_seq(N, dt, 0, 0.3/(np.sqrt(2)))
print(seq)

X = [i*dt for i in range(N)]

plt.plot(X, seq)
plt.show()
