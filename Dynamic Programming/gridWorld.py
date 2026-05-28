# probabilities for left, right, up, down (up is more index)
pi = [0.25, 0.25, 0.25, 0.25]

V = [0] * 16
gamma = 0.9

threshold = 0.1

def move(s, a):
    if a == 0: return s if s % 4 == 0 else s - 1
    elif a == 1: return s if s % 4 == 3 else s + 1
    elif a == 2: return s if s // 4 == 0 else s - 4
    else: return s if s // 4 == 3 else s + 4

def evaluate():
    global V
    V1 = [0.0] * 16

    for state in range(len(V)):
        for action in range(len(pi)):
            for state1 in range(len(V)):
                if state1 == move(state, action):
                    if state == 0 or state == 15: continue
                    else: r = -1

                    V1[state] += pi[action] * (r + gamma * V[state1])

    delta = max([abs(V1[i] - V[i]) for i in range(len(V))])
    V = V1
    return delta

while evaluate() > threshold:
    pass

for x in range(4):
    for y in range(4):
        print(V[y * 4 + x], end = " ")
    print()
