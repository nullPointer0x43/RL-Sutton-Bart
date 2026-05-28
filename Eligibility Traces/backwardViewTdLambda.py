import random

state_n = 19

expected = [(i + 1) / (state_n + 1) for i in range(state_n)]
states = list(range(state_n))
#left = 0; right = 1
actions = [0, 1]
V = {}

gamma = 1
l = [i/10 for i in range(10)]

alpha = [i / 20 for i in range(21)]

def initialize():
    global V

    V = {state: 0.5 for state in states}

def runEpisode(l, a):
    state = 9

    E = {state: 0 for state in states}

    while True:
        action = (1 if random.random() < 0.5 else 0)

        if action == 1:
            next_pos = state + 1
        else:
            next_pos = state - 1

        if next_pos < 0:
            reward = 0
        elif next_pos >= 19:
            reward = 1
        else:
            reward = 0

        E[state] += 1
        v_next = 0 if (next_pos < 0 or next_pos >= 19) else V[next_pos]
        delta = reward + gamma * v_next - V[state]

        for s in states:
            V[s] = V[s] + a * delta * E[s]
            E[s] *= gamma * l

        if next_pos < 0 or next_pos >= 19:
            break

        state = next_pos

def runSim(l, a):
    print("Run: l =", l, " a =", a)
    rms = 0
    for _ in range(100):
        initialize()
        for _ in range(1):
            runEpisode(l, a)
            rms += (sum([(expected[i] - V[i]) ** 2 for i in range(len(expected))]) / len(expected)) ** 0.5

    return rms / 1000

import matplotlib.pyplot as plt

def plot_n_step_performance(l_values, alpha_values, results):
    # Set up the plot aesthetics
    plt.xlabel(r'Step size (alpha)')
    plt.ylabel(r'Average RMS error')
    plt.title(r'n-step TD Performance')

    for i, n_val in enumerate(l_values):
        plt.plot(alpha_values, results[i], label=f'l={n_val}', marker='o')

    # Add legend to identify different n values
    plt.legend()

    # Use a grid for better readability of the U-curves
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.show()

plot_n_step_performance(l, alpha, [[runSim(j, i) for i in alpha] for j in l])