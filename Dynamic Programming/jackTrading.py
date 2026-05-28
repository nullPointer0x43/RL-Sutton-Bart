from scipy.stats import poisson

MAX_CARS = 20
MAX_MOVE = 5

# Pre-calculate Poisson probabilities
def precompute_transitions(lam_rent, lam_ret):
    P = [[0]*(MAX_CARS + 1) for _ in range(MAX_CARS + 1)]
    R = [0]*(MAX_CARS + 1)

    # We limit Poisson to 11 to capture ~99% of probability for lambda 3,4
    for rent in range(12):
        prob_rent = poisson.pmf(rent, lam_rent)
        for ret in range(12):
            prob_ret = poisson.pmf(ret, lam_ret)
            p_combined = prob_rent * prob_ret

            for s in range(MAX_CARS + 1):
                actual_rent = min(s, rent)
                reward = actual_rent * 10
                s_next = min(max(s - actual_rent + ret, 0), MAX_CARS)

                P[s][s_next] += p_combined
                R[s] += p_combined * reward
    return P, R


# Pre-compute for both locations
P1, R1 = precompute_transitions(3, 3)
P2, R2 = precompute_transitions(4, 2)

states = [(s1, s2) for s1 in range(MAX_CARS + 1) for s2 in range(MAX_CARS + 1)]
actions = [a for a in range(-MAX_MOVE, MAX_MOVE + 1)]
pi = {s: {a: 1 / len(actions) for a in actions} for s in states}
gamma = 0.9
V = {s: 0 for s in states}

def expectedReturn(state, action):
    s1, s2 = state

    #action move action number of cars from A to B
    s1 = min(max(s1 - action, 0), MAX_CARS)
    s2 = min(max(s2 + action, 0), MAX_CARS)
    moveCost = abs(action) * 2

    #dayTime reward
    day_reward = R1[s1] + R2[s2] - moveCost

    future_value = 0
    for s11 in range(MAX_CARS + 1):
        for s21 in range(MAX_CARS + 1):
            future_value += P1[s1][s11] * P2[s2][s21] * V[(s11, s21)]

    return day_reward + gamma * future_value

def policyEvaluation():
    def evaluate():
        global V

        V1 = {s: 0 for s in states}

        for state in states:
            for action in actions:
                V1[state] += pi[state][action] * expectedReturn(state, action)

        d = max([abs(V1[i] - V[i]) for i in V])
        V = V1
        return d

    delta = 100
    threshold = 0.1

    while delta > threshold:
        delta = evaluate()

def policyImprovement():
    flag = True

    for state, a1 in pi.items():
        actionRewards = {a: expectedReturn(state, a) for a in a1}

        n = sum([1 for p in actionRewards.values() if p == max(actionRewards.values())])
        improvedActions = {a: (1/n if p == max(actionRewards.values()) else 0)
                           for a, p in actionRewards.items()}

        if improvedActions != pi[state]: flag = False

        pi[state] = improvedActions

    return flag

def policyIteration():
    flag = False #optimal policy is not found
    run = 1

    while not flag:
        print("Run:", run)
        print("Evaluating policy...")
        policyEvaluation()
        print("Improving policy...")
        flag = policyImprovement()
        run += 1

policyIteration()
print("Done!")

print(pi)