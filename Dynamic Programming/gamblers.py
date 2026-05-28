from scipy.stats import poisson

MAXMONEY = 100

states = [s for s in range(MAXMONEY + 1)]
actions = {s: [a for a in range(1, min(s, MAXMONEY - s) + 1)] for s in states}
actions[0] = [0]
actions[100] = [0]
pi = {s: {a: 1 / len(actions[s]) for a in actions[s]} for s in states}

gamma = 1
V = {s: 0 for s in states}
V[100] = 1
ph = 0.3

def expectedReturn(state, action):
    #action taken and the results
    if state + action >= MAXMONEY: winReturns = 1
    else: winReturns = gamma * V[state + action]

    if state - action == 0: loseReturn  = 0
    else: loseReturn = gamma * V[state - action]

    return ph * winReturns + (1 - ph) * loseReturn

def policyEvaluation():
    def evaluate():
        global V

        V1 = {s: 0 for s in states}
        V1[100] = 1

        for state in states:
            for action in actions[state]:
                if state == 0 or state == 100: continue
                V1[state] += pi[state][action] * expectedReturn(state, action)

        d = max([abs(V1[i] - V[i]) for i in V])
        V = V1
        return d

    delta = 100
    threshold = 0.0001

    while delta > threshold:
        delta = evaluate()

def policyImprovement():
    flag = True

    for state, a1 in pi.items():
        actionRewards = {a: expectedReturn(state, a) for a in a1}

        n = sum([1 for a in actionRewards.values() if a == max(actionRewards.values())])
        improvedActions = {a: (1/n if r == max(actionRewards.values()) else 0) for a, r in actionRewards.items()}

        if improvedActions != pi[state]: flag = False

        pi[state] = improvedActions

    return flag

def policyIteration():
    flag = False #optimal policy is not found
    run = 1

    while not flag:
        policyEvaluation()
        flag = policyImprovement()
        run += 1

policyIteration()
print(pi)