from lib.oneqrb import *
import qecc as q

delta_t = 1000
t_slice = np.linspace(0, np.pi/2, delta_t + 1)
delta = 0.01    # noise angle

x_pi2 = I_1q
hx = 1 / 2 * X_1q
hz = delta / 2 / (np.pi / 2) * Z_1q
h = hx + hz
for t in t_slice[1:]:
    x_pi2 = np.dot(expm(-1j * h * t_slice[1]), x_pi2)

delta_tilda = delta/(np.pi/2)
omega = np.sqrt(1+delta_tilda**2)
x_pi2_thr = np.cos(np.pi/4*omega) * I_1q - 1j * np.sin(np.pi/4*omega) * 1/omega * (X_1q + delta_tilda*Z_1q)

print(gate_fidelity_1q(x_pi2, x_pi2_thr))