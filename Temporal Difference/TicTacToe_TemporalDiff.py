import random

#value mapping for states
V = {(0, ) * 9: 0.5}
a = 0.1
p = 0.1

# x0, x1, x2
# x3, x4, x5
# x6, x7, x8
# 0 -> unknown
# 1 -> O
# 2 -> X -> player

#winChecker
def winChecker(state):
    # columns check
    for i in range(3):
        if state[i] == state[i + 3] and state[i + 3] == state[i + 6] and state[i] != 0:
            return state[i] - 1

    # rows check
    for i in range(3):
        if state[i * 3] == state[i * 3 + 1] and state[i * 3 + 1] == state[i * 3 + 2] and state[i * 3] != 0:
            return state[i * 3] - 1

    # primary diagonal check
    if state[0] == state[4] and state[4] == state[8] and state[0] != 0:
        return state[0] - 1

    # secondary diagonal check
    if state[2] == state[4] and state[4] == state[6] and state[2] != 0:
        return state[2] - 1

    if 0 not in state:
        return 0

    # return 0.5 if noone is winning
    # returns 1 if X wins, 0 if O wins
    return 0.5

def playerMove(state):
    legal = []

    for position in range(9):
        if state[position] == 0:
            move = state[:position] + (2, ) + state[position + 1:]
            legal.append(move)
            V.setdefault(move, winChecker(move))

    if random.random() < p:
        return random.choice(legal)
    else:
        greedy_state = legal[0]

        for choice in legal:
            if V[choice] > V[greedy_state]:
                greedy_state = choice

        return greedy_state


def opponentMove(state):
    legal = []

    for position in range(9):
        if state[position] == 0:
            move = state[:position] + (1, ) + state[position + 1:]
            legal.append(move)

    move = random.choice(legal)
    V.setdefault(move, winChecker(move))
    return move

def move(state):
    s1 = playerMove(state)

    if winChecker(s1) != 0.5:
        V[state] = V[state] + a * (V[s1] - V[state])
        return s1
    else:
        s2 = opponentMove(s1)
        V[state] = V[state] + a * (V[s2] - V[state])
        return s2


# loop:
# search for possible next steps and their values
# if no look up for next steps choose any step (exploration) and repeat
# if look up exists find maximum value and take that step -> update value to be close to the new state value
# randomly choose exploratory states

for i in range(10000): #play through 10000 games
    state = (0, ) * 9

    while winChecker(state) == 0.5:
        state = move(state)

    if i % 500:
        print(f"Total states explored: {len(V)}")
        print(f"Value of starting board: {V[(0, 0, 0, 0, 0, 0, 0, 0, 0)]}")

# the following code is AI generated, idk didnt feel like writing an entire game loop to test out if it learnt
def print_board(state):
    symbols = {0: ".", 1: "O", 2: "X"}
    for i in range(0, 9, 3):
        print(f"{symbols[state[i]]} {symbols[state[i + 1]]} {symbols[state[i + 2]]}")
    print()


# The Play Loop
state = (0,) * 9
print("Game Start! You are O (1), AI is X (2).")
print("Positions are 0-8 (top-left to bottom-right).")

while winChecker(state) == 0.5:
    print_board(state)

    # Human Turn
    try:
        move_idx = int(input("Enter your move (0-8): "))
        if state[move_idx] != 0:
            print("Invalid move, spot taken! Try again.")
            continue
    except (ValueError, IndexError):
        print("Please enter a number between 0 and 8.")
        continue

    # Update state with Human move (1)
    state = state[:move_idx] + (1,) + state[move_idx + 1:]

    if winChecker(state) != 0.5:
        break

    # AI Turn
    # We use your playerMove function here
    print("AI is thinking...")
    state = playerMove(state)

# Final Result
print_board(state)
result = winChecker(state)
if result == 1:
    print("AI (X) Wins! Better luck next time.")
elif result == 0 and 0 not in state:
    print("It's a Draw!")
else:
    print("You (O) Won! You beat the machine.")