import random

bandits = []
n = 10

minVal = -10
maxVal = 10

noise_var = 5
Q1 = 0
alpha = 0.1
epsilon = 0

QMap = []
counts = []

def initialize():
    global bandits, QMap, counts

    bandits = [random.randint(minVal, maxVal) for _ in range(n)]
    QMap = [Q1 for _ in range(n)]
    counts = [0 for _ in range(n)]

def getVal(bandit):
    return bandits[bandit] + random.random() * noise_var

def iteration():
    global bandits

    if random.random() < epsilon:
        # exploration
        index = random.randint(0, n - 1)
    else:
        # greedy
        indices = [i for i, num in enumerate(QMap) if num == max(QMap)]
        index = random.choice(indices)

    value = getVal(index)

    # fixed alpha
    QMap[index] = QMap[index] + alpha * (value - QMap[index])

    # varying alpha
    counts[index] += 1
    #QMap[index] = QMap[index] + (value - QMap[index]) / counts[index]

    return value

def updateBandits():
    global bandits
    bandits = [i + (random.random() - 0.5) * 2 for i in bandits]

data_collected = [0] * 1000
for i in range(2000):#average of 2000 runs
    initialize()

    for j in range(1000):#1000 steps
        data_collected[j] = (data_collected[j] * i + iteration()) / (i + 1)
        updateBandits()

print(data_collected)