import random

states = [(x, y) for x in range(10) for y in range(7)]
# 0 1 2
# 3 8 4
# 5 6 7
actions = list(range(8)) # 0 -> L ; 1 -> R ; 2 -> D ; 3 -> U

Q = {(state, action): 0.0 for state in states for action in actions}
wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

policy = {state: {action: 1/len(actions) for action in actions} for state in states}
alpha = 0.3
epsilon = 0.3

start = (0, 3)
target = (7, 3)


def move(state, action):
    x, y = state

    if action in [0, 3, 5]: x = max(state[0] - 1, 0)
    elif action in [2, 4, 7]: x = min(state[0] + 1, 9)
    elif action in [0, 1, 2]: y = max(state[1] - 1, 0)
    elif action in [5, 6, 7]: y = min(state[1] + 1, 6)

    next_state = (x, min(y + wind[x], 6))
    return next_state

def reward(state, action):
    if move(state, action) == target: return 0
    return -2

def runEpisode():
    global policy

    state = start

    action = random.choices(actions, weights = list(policy[state].values()), k = 1)[0]

    for _ in range(200):
        s1 = move(state, action)
        r = reward(state, action)
        a1 = random.choices(actions, weights = list(policy[s1].values()), k = 1)[0]

        Q[(state, action)] = Q[(state, action)] + alpha * (r + Q[(s1, a1)] - Q[(state, action)])

        state = s1
        action = a1

        if state == target: break

    policy = {state: {action: epsilon / len(actions) for action in actions} for state in states}

    for state in policy:
        legalStateActions = [(state, action) for action in actions]
        rewards = [Q[i] for i in legalStateActions]
        legalActions = [i[1] for i in legalStateActions if Q[i] == max(rewards)]

        p = 1 - (len(actions) - len(legalActions)) * epsilon / len(actions)

        for action in legalActions:
            policy[state][action] = p / len(legalActions)

for i in range(50000):
    runEpisode()
    if i % 5000 == 0: print(i)

V = {state: sum([policy[state][action] * Q[(state, action)] for action in actions]) for state in states}


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

# We use a limit (e.g., 50) to prevent infinite loops if the policy hasn't converged
for _ in range(50):
    if current_state == target:
        break

    # Pick the best action according to the learned Q-values
    best_action = max(actions, key=lambda a: Q[(current_state, a)])

    # Move to the next state
    current_state = move(current_state, best_action)
    trajectory.append(current_state)

plot_policy(trajectory)