import random
import math

bandits = []
n = 10

minVal = -10
maxVal = 10

noise_var = 5
alpha = 0.01
R_mean = 0
t = 0

HMap = []

def choose(probs):
    chosen_action = random.choices(list(range(n)), weights=probs, k=1)[0]
    return chosen_action

def initialize():
    global bandits, HMap, R_mean, t

    bandits = [random.randint(minVal, maxVal) for _ in range(n)]
    HMap = [0 for _ in range(n)]
    R_mean = 0
    t = 0

def getVal(bandit):
    return bandits[bandit] + random.random() * noise_var

def iteration():
    global bandits, HMap, R_mean, t

    exponents = [math.exp(i) for i in HMap]
    probabilities = [i / sum(exponents) for i in exponents]

    choice = choose(probabilities)
    value = getVal(choice)

    for key in range(len(HMap)):
        if key == choice:
            HMap[key] += alpha * (value - R_mean) * (1 - probabilities[choice])
        else:
            HMap[key] -= alpha * (value - R_mean) * probabilities[choice]

    R_mean = (R_mean * t + value) / (t + 1)
    t += 1

    return value

data_collected = [0] * 1000
for i in range(2000):#average of 2000 runs
    initialize()

    for j in range(1000):#1000 steps
        data_collected[j] = (data_collected[j] * i + iteration()) / (i + 1)

print(data_collected)