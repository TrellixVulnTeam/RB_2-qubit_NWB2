from lib.oneqrb import *

delta_list = [x * 0.01 for x in list(range(1, 51))]

delta_t = 100
t_slice = np.linspace(0, np.pi/2, delta_t + 1)


F_tr = []

for delta in delta_list:
    x_pi2 = I_1q
    hx = 1/2 * X_1q
    hz = (delta/2) / (np.pi/2) * 1/2 * Z_1q
    h = hx + hz
    for t in t_slice[1:]:
        x_pi2 = np.dot(expm(-1j * h * t_slice[1]), x_pi2)
    F_tr.append(gate_fidelity_1q(x_pi2, X_pi_2))

print(F_tr)

f5 = open('const_delta_pulse_tr_fidelity_h_1q.pkl', 'wb')
pickle.dump(F_tr, f5)
f5.close()