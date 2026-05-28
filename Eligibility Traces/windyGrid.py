import random
import numpy as np

# Grid 10x7
width, height = 10, 7
states = [(x, y) for x in range(width) for y in range(height)]
actions = list(range(9))  # King's moves + Stand still
wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

Q = np.zeros((width, height, 9))
alpha = 0.3
epsilon = 0.1  # Standard epsilon
gamma = 1.0
lam = 0.8
start, target = (0, 3), (7, 3)


def move(state, action):
    x, y = state
    # King's Moves Mapping
    dx = [-1, 0, 1, -1, 1, -1, 0, 1, 0]
    dy = [-1, -1, -1, 0, 0, 1, 1, 1, 0]

    new_x = max(0, min(x + dx[action], width - 1))
    # Wind only affects the y-coordinate based on the x-position
    new_y = max(0, min(y + dy[action] + wind[new_x], height - 1))
    return (new_x, new_y)


def choose_action(state):
    if random.random() < epsilon:
        return random.choice(actions)
    return np.argmax(Q[state[0], state[1], :])


def run_episode():
    state = start
    action = choose_action(state)
    E = np.zeros((width, height, 9))

    while state != target:
        next_state = move(state, action)
        next_action = choose_action(next_state)
        reward = -1

        delta = reward + gamma * Q[next_state[0], next_state[1], next_action] - Q[state[0], state[1], action]

        # Accumulating traces
        E[state[0], state[1], action] += 1

        # Vectorized update is much faster in Python/NumPy
        Q[:] += alpha * delta * E
        E[:] *= gamma * lam

        state, action = next_state, next_action


# Training
for i in range(1000):
    run_episode()
    if i % 50 == 0: print(f"Episode {i} complete")

import matplotlib.pyplot as plt

MAX_CARS = 20
MAX_MOVE = 5

def plot_policy(trajectory):
    global V
    policy_grid = [[0]*10 for _ in range(7)]

    for (s1, s2), actions_dict in V.items():
        policy_grid[s2][s1] = V[(s1, s2)]

    # Create the heatmap
    plt.figure(figsize=(8, 6))
    # 'RdBu' is great because it shows negative moves in Red and positive in Blue
    im = plt.imshow(policy_grid, origin='lower', cmap='RdBu')

    plt.title("Windy grid problem Kings moves + stand still\n"+
              "Trajectory from Start to Target:"+
              " -> ".join([str(s) for s in trajectory])+
              f"\nlength of trajectory: {len(trajectory)}", wrap = True)

    # Optional: Add grid lines for clarity
    plt.xticks(range(10))
    plt.yticks(range(7))
    plt.grid(which='both', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    plt.show()


# Follow the learned policy from start to target
current_state = start
trajectory = [current_state]
V = {(x, y): max([Q[x, y, action] for action in actions]) for (x, y) in states}

plot_policy([])