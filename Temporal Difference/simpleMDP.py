import random

expected = [1/6, 2/6, 3/6, 4/6, 5/6]
#A -> 0, B -> 1, C -> 2, D -> 3, E -> 4 final_blocking = 5
states = list(range(6))
#left = 0; right = 1
actions = [0, 1]
nextStates = {}
alpha = 0.15
V = {state: 0 for state in states}
rewards = {(state, action): (1 if state == 4 and action == 1 else 0) for state in states for action in actions}

for state in range(5):
    for action in range(2):
        if state == 0 and action == 0: ns = 5
        elif state == 4 and action == 1: ns = 5
        elif state == 5: ns = 5
        else: ns = state + (action - 0.5) * 2

        nextStates[(state, action)] = ns

def runEpisode():
    state = 2

    for _ in range(100):
        action = 1 if random.random() < 0.5 else 0
        ns = nextStates[(state, action)]

        V[state] = V[state] + alpha * (V[ns] + rewards[(state, action)] - V[state])
        if ns == 5: break

        state = ns

error = []

for i in range(10000):
    rms = (sum([(expected[i]-V[i])**2 for i in range(len(expected))])/len(expected))**0.5
    error.append(rms)
    runEpisode()

import matplotlib.pyplot as plt

# 2. Create the plot
plt.plot(error, linestyle='-', color='b', label='Performance')

plt.title('TD')
plt.xlabel('Episodes x 100')
plt.ylabel('RMSE')

plt.grid(True)
plt.legend()
plt.show()