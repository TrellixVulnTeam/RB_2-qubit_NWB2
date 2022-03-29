from lib.oneqrb import *
import matplotlib.pyplot as plt

cliff_seq = np.random.choice(24, 50, replace=True)
data_point = np.arange(1, 50, 1, dtype=int)
n = 100  # number of sample of noises
dephasing_angle = 0.5

fidelity_channel = np.zeros(len(data_point))
fidelity_hamiltonian = np.zeros(len(data_point))

for i in range(n):
    # noise_seq = np.random.normal(0, dephasing_angle) * np.ones(len(cliff_seq))  # static noise
    # noise_seq = np.random.normal(0, dephasing_angle, len(cliff_seq))  # uncorrelated noise
    noise_seq = dephasing_angle * np.ones(len(cliff_seq))  # constant noise

    # channel_seq = get_seq_1q(cliff_seq, noise_seq, delta_t=100, noise_type=CHANNEL_NOISE)
    # hamiltonian_seq = get_seq_1q(cliff_seq, noise_seq, delta_t=100, noise_type=HAMILTONIAN_NOISE)

    # noise_seq2 = np.random.normal(0, dephasing_angle/np.sqrt(2), (len(cliff_seq), 2))   # uncorrelated noise for Hamiltonian
    # noise_seq1 = noise_seq2.sum(axis=1)  # uncorrelated noise for channel

    channel_seq = get_seq_1q(cliff_seq, noise_seq, delta_t=100, noise_type=CHANNEL_NOISE)
    hamiltonian_seq = get_seq_1q(cliff_seq, noise_seq, delta_t=100, noise_type=HAMILTONIAN_NOISE)
    # hamiltonian_seq = get_seq_1q_h_diff_pulse_noise(cliff_seq, noise_seq2, delta_t=100)

    # initialize sequence S
    S_channel = I_1q
    S_hamiltonian = I_1q
    for l in range(len(cliff_seq)):
        S_channel = channel_seq[l] @ S_channel
        S_hamiltonian = hamiltonian_seq[l] @ S_hamiltonian
        if (l+1) in data_point:
            S_perfect = get_perfect_seq_1q(cliff_seq[:(l+1)])
            idx = np.where(data_point == (l+1))[0][0]
            fidelity_channel[idx] += gate_fidelity_1q(S_channel, S_perfect)/n
            fidelity_hamiltonian[idx] += gate_fidelity_1q(S_hamiltonian, S_perfect)/n

plt.plot(data_point, fidelity_channel, 'ro', markersize=2, label='channel noise')
plt.plot(data_point, fidelity_hamiltonian, 'bo', markersize=2, label='Hamiltonian noise')

# plt.title('uncorrelated dephasing angle')
# plt.title('static dephasing angle')
plt.title('constant dephasing angle')

plt.xlabel("sequence length")
plt.ylabel("trace fidelity")
plt.legend()
plt.show()
