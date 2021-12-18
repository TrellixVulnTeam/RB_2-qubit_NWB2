from lib.nswrb import *
import matplotlib.pyplot as plt

wav = np.zeros((4, 100))
std = [10, 20, 30, 40]
dt = 1e-5
h = time_varying_gaussian_noise(wav, dt, std, f_noise=1/(10*dt))

for i in range(len(h)):
    print(h[i])
