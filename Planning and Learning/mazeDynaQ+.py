import random
import pygame

# GUI Setup
pygame.init()

WIDTH, HEIGHT = 600, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))

N = 10
CELL_SIZE = min(WIDTH // N, HEIGHT // N)

map1 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 3], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 2, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
map2 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 3], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 2, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
grid = map2

heatmap = [[0] * N for _ in range(N)]

pressed = False

mode = 1

# Dyna Q Setup
states = []
actions = []

start, target = (), ()
state = ()

Q = {}
T = {}

model = {}
env = 0

gamma = 0.95
alpha = 0.1
epsilon = 0.1
k = 0.001

n = 5
n_averaging = 30

def initialize():
    global states, actions, Q, start, target, state, T

    states = [(x, y) for x in range(N) for y in range(N)]
    actions = list(range(4))
    Q = {(state, action): 0 for state in states for action in actions}
    T = {(state, action): 0 for state in states for action in actions}

    start = [(x, y) for x in range(N) for y in range(N) if grid[y][x] == 2][0]
    target = [(x, y) for x in range(N) for y in range(N) if grid[y][x] == 3][0]

    state = start

def executeA(S, A):
    if A == 0: S1 = (S[0] - 1, S[1])
    elif A == 1: S1 = (S[0] + 1, S[1])
    elif A == 2: S1 = (S[0], S[1] - 1)
    else: S1 = (S[0], S[1] + 1)

    if not (0 <= S1[1] < N and 0 <= S1[0] < N) or grid[S1[1]][S1[0]] == 1: S1 = S

    if S1 == target: return S1, 1

    return S1, 0

def getAction(S):
    if random.random() < epsilon:
        return random.choice(actions)

    state_action_values = [Q[S, action] for action in actions]
    return random.choice([action for action in actions if Q[S, action] == max(state_action_values)])

def planning():
    global T

    if model:
        for _ in range(n):
            S, A = random.choice(list(model.keys()))
            S1, R = model[(S, A)]

            maxNextVal = max([Q[S1, a] for a in actions])
            Q[(S, A)] = Q[(S, A)] + alpha * (R + k * T[(S, A)]**0.5 + gamma * maxNextVal - Q[(S, A)])

def runMove(S):
    global T
    A = getAction(S)
    S1, R = executeA(S, A)

    # Direct RL
    maxNextVal = max([Q[S1, a] for a in actions])

    Q[(S, A)] = Q[(S, A)] + alpha * (R + gamma * maxNextVal - Q[(S, A)])

    T = {z: T[z] + 1 for z in T}
    T[(S, A)] = 0

    # Model Learning
    model[(S, A)] = (S1, R)

    # Planning
    planning()

    return S1

def makeHeatMap():
    global heatmap

    maxVal = max([Q[((0, 0), action)] for action in actions])
    minVal = max([Q[((0, 0), action)] for action in actions])

    for y in range(N):
        for x in range(N):
            if grid[y][x] != 1:
                heatmap[y][x] = max([Q[((x, y), action)] for action in actions])
                maxVal = max(maxVal, heatmap[y][x])
                minVal = min(minVal, heatmap[y][x])

    for y in range(N):
        for x in range(N):
            if grid[y][x] != 1:
                heatmap[y][x] -= minVal
                heatmap[y][x] /= (maxVal - minVal)

def masterDraw():
    screen.fill((255, 255, 255))

    for y in range(N):
        for x in range(N):
            rect = (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)

            # Base Layer: Terrain
            if grid[y][x] == 1:
                pygame.draw.rect(screen, (40, 40, 40), rect)
            elif grid[y][x] == 2:
                pygame.draw.rect(screen, (200, 255, 200), rect)  # Start
            elif grid[y][x] == 3:
                pygame.draw.rect(screen, (255, 200, 200), rect)  # Finish
            else:
                if mode == 4:
                    pygame.draw.rect(screen, (0, 0, 255 * heatmap[y][x]), rect)

    for i in range(N + 1):
        pygame.draw.line(screen, (220, 220, 220), (i * CELL_SIZE, 0), (i * CELL_SIZE, HEIGHT))
        pygame.draw.line(screen, (220, 220, 220), (0, i * CELL_SIZE), (WIDTH, i * CELL_SIZE))

    pygame.display.flip()

moves = 0
revisions = 0
line = []

running = True
while running:
    # Event Handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            if mode < 4:
                mode += 1
                if mode == 4:
                    print(grid)
                    initialize()
            elif mode == 4:
                env = (env + 1) % 2

                if env:
                    grid = map1
                else:
                    grid = map2


    # Check mouse state outside the event loop
    mouse_buttons = pygame.mouse.get_pressed()
    if mouse_buttons[0]:
        mouse_pos = pygame.mouse.get_pos()
        x = mouse_pos[0] // CELL_SIZE
        y = mouse_pos[1] // CELL_SIZE

        if 0 <= x < N and 0 <= y < N:
            grid[y][x] = mode

    if mode == 4:
        state = runMove(state)
        moves += 1

        if state == target:
            state = start
            revisions += 1

            if revisions % n_averaging == 0:
                makeHeatMap()
                print("Episode over in: ", moves / n_averaging)
                line.append(moves / n_averaging)

                moves = 0
                revisions = 0

    # Draw
    masterDraw()
pygame.quit()
