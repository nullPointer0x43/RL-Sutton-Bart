import random

state_n = 19

expected = [(i + 1) / (state_n + 1) for i in range(state_n)]
states = list(range(state_n))
#left = 0; right = 1
actions = [0, 1]
V = {}
gamma = 1

n = [2**i for i in range(10)]
alpha = [i * 1 / 20 for i in range(20)]

def initialize():
    global V

    V = {state: 0.5 for state in states}

def runEpisode(n, a):
    state = 9

    t = 0
    T = 100
    epStates = [state]
    epRewards = [0]

    while True:
        if t < T:
            action = (1 if random.random() < 0.5 else 0)

            current_pos = epStates[-1]
            if action == 1:
                next_pos = current_pos + 1
            else:
                next_pos = current_pos - 1

            if next_pos < 0:
                reward = 0
                T = t + 1
            elif next_pos >= 19:
                reward = 1
                T = t + 1
            else:
                reward = 0

            epRewards.append(reward)
            epStates.append(next_pos)

        tau = t - n + 1

        if tau >= 0:
            G = 0
            for i in range(tau + 1, min(tau + n, T) + 1):
                G += (gamma ** (i - tau - 1)) * epRewards[i]

            if (tau + n) < T:
                G += (gamma ** n) * V[epStates[tau + n]]

            state = epStates[tau]

            V[state] = V[state] + a * (G - V[state])

        if tau == T - 1:
            break

        t += 1

def runSim(n, a):
    print("Run: n =", n, "a =", a)
    rms = 0
    for _ in range(100):
        initialize()
        for _ in range(10):
            runEpisode(n, a)
            rms += (sum([(expected[i] - V[i]) ** 2 for i in range(len(expected))]) / len(expected)) ** 0.5

    return rms / 1000

lines = [[runSim(i, a) for a in alpha] for i in n]

import matplotlib.pyplot as plt

def plot_n_step_performance(n_values, alpha_values, results):
    # Set up the plot aesthetics
    plt.xlabel(r'Step size (alpha)')
    plt.ylabel(r'Average RMS error')
    plt.title(r'n-step TD Performance')

    # Iterate through each n and its corresponding result line
    for i, n_val in enumerate(n_values):
        plt.plot(alpha_values, results[i], label=f'n={n_val}', marker='o')

    # Add legend to identify different n values
    plt.legend()

    # Use a grid for better readability of the U-curves
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.show()

plot_n_step_performance(n, alpha, lines)