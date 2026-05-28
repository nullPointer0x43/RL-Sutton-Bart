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

policy = {s: {1: 1, 0: 0} for s in states}
V = {}

for dealer_card in range(1, 11):
    for usable_ace in range(2):
        policy[(20, dealer_card, usable_ace)] = {1: 0, 0: 1}
        policy[(21, dealer_card, usable_ace)] = {1: 0, 0: 1}

results = {(s, a): [] for s in states for a in actions}

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

def runEpisode():
    global dealer, player, results

    runStates = []
    initialize()
    explorativeStart = True

    # player plays according to its policy
    while True:
        p_sum, usable_ace = get_hand_info(player)

        if p_sum > 21: break
        elif p_sum < 12: action = 1
        else:
            state = (p_sum, min(dealer[0], 10), usable_ace)

            if explorativeStart:
                action = random.choice(actions)
                explorativeStart = False
            else:
                legalActions = [a for a in policy[state] if policy[state][a] == max(policy[state].values())]
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

    for state in set(runStates):
        results[state].append(result)

def policyEvaluation():
    global V

    n = 50000
    for i in range(n):
        runEpisode()

    V = {(s, a): sum(results[(s, a)]) / len(results[(s, a)])
    for s, a in results if len(results[(s, a)]) > 0}

def policyImprovement():
    flag = True

    for state, a1 in policy.items():
        actionRewards = {a: V.get((state, a), 0) for a in a1}

        n = sum([1 for a in actionRewards.values() if a == max(actionRewards.values())])
        improvedActions = {a: (1/n if r == max(actionRewards.values()) else 0) for a, r in actionRewards.items()}

        if improvedActions != policy[state]: flag = False

        policy[state] = improvedActions

    return flag

optimal = False
run = 0

while not optimal:
    print("Run:", run)
    print("Evaluating...")
    policyEvaluation()
    print("Improving...")
    optimal = policyImprovement()
    run += 1

print("Done!")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_policy(policy_dict):
    """
    Visualizes the policy dictionary as two heatmaps.
    S = Stick (0) -> usually shown as a darker/cooler color
    H = Hit (1) -> usually shown as a brighter/warmer color
    """
    # Define ranges for the axes
    player_sums = np.arange(12, 22)
    dealer_cards = np.arange(1, 11)

    # Initialize grids for the two scenarios
    # 0 = Usable Ace, 1 = No Usable Ace (matching your dict key format)
    grid_ace = np.zeros((len(player_sums), len(dealer_cards)))
    grid_no_ace = np.zeros((len(player_sums), len(dealer_cards)))

    for i, p in enumerate(player_sums):
        for j, d in enumerate(dealer_cards):
            # Extract the best action (the one with the highest probability)
            # Usable Ace (1)
            state_ace = (p, d, 1)
            if state_ace in policy_dict:
                actions = policy_dict[state_ace]
                grid_ace[i, j] = max(actions, key=actions.get)

            # No Usable Ace (0)
            state_no_ace = (p, d, 0)
            if state_no_ace in policy_dict:
                actions = policy_dict[state_no_ace]
                grid_no_ace[i, j] = max(actions, key=actions.get)

    # Set up the visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot Usable Ace
    sns.heatmap(grid_ace, annot=True, cbar=False, xticklabels=dealer_cards,
                yticklabels=player_sums, cmap="Accent_r", ax=ax1)
    ax1.set_title('Policy: Usable Ace (0=Stick, 1=Hit)')
    ax1.set_xlabel('Dealer Showing')
    ax1.set_ylabel('Player Sum')
    ax1.invert_yaxis()

    # Plot No Usable Ace
    sns.heatmap(grid_no_ace, annot=True, cbar=False, xticklabels=dealer_cards,
                yticklabels=player_sums, cmap="Accent_r", ax=ax2)
    ax2.set_title('Policy: No Usable Ace (0=Stick, 1=Hit)')
    ax2.set_xlabel('Dealer Showing')
    ax2.set_ylabel('Player Sum')
    ax2.invert_yaxis()

    plt.tight_layout()
    plt.show()

# Example usage with your specific dictionary format:
plot_policy(policy)