import random

dealer = []
player = []
actions = [0, 1] # 0 -> stick, 1 -> hit

# defined by current sum of player 12-21 (anything less implies he can always hit and then follow strategy
# card shown by the dealer (A, 2, 3, 4, 5, 6, 7, 8, 9, 10) all face cards also count as 10
# if they have a usable ace or not
states = [(player_sum, dealer_card, usable_ace)
          for player_sum in range(12, 22)
          for dealer_card in range(1, 11)
          for usable_ace in range(2)]

# target policy
policy = {s: {1: 1, 0: 0} for s in states}

for dealer_card in range(1, 11):
    for usable_ace in range(2):
        policy[(20, dealer_card, usable_ace)] = {1: 0, 0: 1}
        policy[(21, dealer_card, usable_ace)] = {1: 0, 0: 1}

results = {s: [] for s in states}

# random off policy
mu = {s: {1: 0.5, 0: 0.5} for s in states}

def initialize():
    global dealer, player
    dealer = [random.randint(1, 13) for _ in range(2)]
    player = [random.randint(1, 13) for _ in range(2)]


def get_hand_info(hand):
    values = [min(card, 10) for card in hand]
    current_sum = sum(values)

    # Check if there's an Ace (1) and if we can add 10 to make it an 11
    usable_ace = (1 if 1 in values and current_sum + 10 <= 21 else 0)

    if usable_ace:
        current_sum += 10

    return current_sum, usable_ace

def importanceSampling(trajectory, state):
    start = trajectory.index(state)
    rho = 1

    for i in range(start, len(trajectory)):
        rho *= policy[trajectory[i][0]][trajectory[i][1]] / mu[trajectory[i][0]][trajectory[i][1]]

    return rho

def runEpisode():
    global dealer, player, results

    runStates = []
    initialize()

    # player plays according to its policy
    while True:
        p_sum, usable_ace = get_hand_info(player)

        if p_sum > 21: break
        elif p_sum < 12: action = 1
        else:
            state = (p_sum, min(dealer[0], 10), usable_ace)

            legalActions = [a for a in mu[state] if mu[state][a] == max(mu[state].values())]
            action = random.choice(legalActions)
            runStates.append((state, action))

        if action == 0: break
        else:
            player.append(random.randint(1, 13))

            p_sum, _ = get_hand_info(player)
            if p_sum > 21: break

    while True:
        d_sum, _ = get_hand_info(dealer)
        if d_sum >= 17: break
        dealer.append(random.randint(1, 13))

    p_final, _ = get_hand_info(player)
    d_final, _ = get_hand_info(dealer)

    if p_final > 21: result = -1
    elif d_final > 21: result = 1
    elif p_final == d_final: result = 0
    elif p_final > d_final: result = 1
    else: result = -1

    # given the trajectory
    for state_action in set(runStates):
        # instead of just G(t) make it rho * G(t)
        results[state_action[0]].append(importanceSampling(runStates, state_action) * result)

n = 10000000
for i in range(n):
    if i % 500000 == 0: print("Run", i)
    runEpisode()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_blackjack_value(v_dict_ace, v_dict_no_ace):
    """
    Inputs:
    - v_dict_ace: Dictionary {(player_sum, dealer_card): value} for usable ace
    - v_dict_no_ace: Dictionary {(player_sum, dealer_card): value} for no usable ace
    """

    # Define the axes ranges
    player_sums = range(12, 22)
    dealer_cards = range(1, 11)

    # Function to convert dictionary to 2D numpy array
    def dict_to_grid(v_dict):
        grid = np.zeros((len(player_sums), len(dealer_cards)))
        for i, p in enumerate(player_sums):
            for j, d in enumerate(dealer_cards):
                # Using .get() to handle any missing states safely
                grid[i, j] = v_dict.get((p, d), 0)
        return grid

    # Prepare the grids
    grid_ace = dict_to_grid(v_dict_ace)
    grid_no_ace = dict_to_grid(v_dict_no_ace)

    # Setup the plot
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Plot Usable Ace
    sns.heatmap(grid_ace, annot=True, fmt=".2f",
                xticklabels=dealer_cards, yticklabels=player_sums,
                cmap='RdYlGn', ax=axes[0], vmin=-1, vmax=1)
    axes[0].set_title('Value Function (Usable Ace)', fontsize=15)
    axes[0].set_xlabel('Dealer Showing Card', fontsize=12)
    axes[0].set_ylabel('Player Current Sum', fontsize=12)
    axes[0].invert_yaxis()  # To show 21 at the top if preferred

    # Plot No Usable Ace
    sns.heatmap(grid_no_ace, annot=True, fmt=".2f",
                xticklabels=dealer_cards, yticklabels=player_sums,
                cmap='RdYlGn', ax=axes[1], vmin=-1, vmax=1)
    axes[1].set_title('Value Function (No Usable Ace)', fontsize=15)
    axes[1].set_xlabel('Dealer Showing Card', fontsize=12)
    axes[1].set_ylabel('Player Current Sum', fontsize=12)
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.show()

V_ace = {
    (s[0], s[1]): sum(results[s]) / len(results[s])
    for s in results if s[2] == True and len(results[s]) > 0
}
V_no_ace = {
    (s[0], s[1]): sum(results[s]) / len(results[s])
    for s in results if s[2] == False and len(results[s]) > 0
}
plot_blackjack_value(V_ace, V_no_ace)