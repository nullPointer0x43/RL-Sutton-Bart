import random
import pygame
import collections
import math

# GUI Setup
pygame.init()
WIDTH, HEIGHT = 600, 600
N = 15
CELL_SIZE = min(WIDTH // N, HEIGHT // N)
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
grid = [[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 3], [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3], [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3], [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3], [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1]]
pressed = False
mode = 1
heatmap = [[0] * N for _ in range(N)]
history = collections.deque(maxlen=1000)
success_count = 0

# Monte Carlo Setup
states = []
returns = {}
actions = [(i, j) for i in range(-1, 2) for j in range(-1, 2)]
Q = {}
epsilon = 0.3
gamma = 0.99


def get_all_start_trajectories():
    all_paths = []
    # Find every cell marked as a Start (2)
    starts = [(x, y) for y in range(N) for x in range(N) if grid[y][x] == 2]

    for start_node in starts:
        # Initial state: (x, y, 0, 0)
        state = (start_node[0], start_node[1], 0, 0)
        path = [state]

        # Simulate until finish, crash, or timeout
        for _ in range(100):
            # Greedy action selection
            q_vals = [Q.get((state, a), -1000) for a in range(len(actions))]
            best_action_idx = q_vals.index(max(q_vals))
            ax, ay = actions[best_action_idx]

            # Physics (Sutton-Barto style, deterministic for plotting)
            vx_new = max(0, min(state[2] + ax, 4))
            vy_new = max(0, min(state[3] + ay, 4))

            # Maintain minimum movement if possible
            if vx_new == 0 and vy_new == 0:
                vx_new, vy_new = (state[2], state[3]) if state[2] + state[3] > 0 else (1, 0)

            nx, ny = state[0] + vx_new, state[1] - vy_new

            # Stop if out of bounds or wall
            if nx < 0 or nx >= N or ny < 0 or ny >= N or grid[ny][nx] == 1:
                break

            state = (nx, ny, vx_new, vy_new)
            path.append(state)

            if grid[ny][nx] == 3:  # Success!
                break

        all_paths.append(path)
    return all_paths

def update_heatmap():
    global heatmap
    for y in range(N):
        for x in range(N):
            if grid[y][x] != 0: continue

            # ONLY look at states where we have actually recorded returns
            pixel_vals = [Q[(s, a)] for (s, a) in Q
                          if s[0] == x and s[1] == y and returns.get((s, a), 0) > 0]

            if pixel_vals:
                heatmap[y][x] = max(pixel_vals)
            else:
                heatmap[y][x] = -500

def master_draw():
    screen.fill((255, 255, 255))
    for y in range(N):
        for x in range(N):
            rect = (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            center = (x * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + CELL_SIZE // 2)

            # Base Layer: Terrain
            if grid[y][x] == 1:
                pygame.draw.rect(screen, (40, 40, 40), rect)  # Walls
            elif grid[y][x] == 2:
                pygame.draw.rect(screen, (200, 255, 200), rect)  # Start
            elif grid[y][x] == 3:
                pygame.draw.rect(screen, (255, 200, 200), rect)  # Finish

            # Top Layer: Policy Arrows (Mode 4)
            if mode == 4 and grid[y][x] != 1:
                trajectories = get_all_start_trajectories()

                for path in trajectories:
                    if len(path) > 1:
                        # Convert grid coords to screen pixels
                        points = [(p[0] * CELL_SIZE + CELL_SIZE // 2,
                                   p[1] * CELL_SIZE + CELL_SIZE // 2) for p in path]

                        # Draw the path line
                        # We use a semi-transparent looking color or thin lines
                        # to see the 'funnel' effect toward the finish
                        pygame.draw.lines(screen, (255, 50, 50), False, points, 2)

                        # Draw a small dot at the end of each successful path
                        if grid[path[-1][1]][path[-1][0]] == 3:
                            pygame.draw.circle(screen, (0, 0, 255), points[-1], 3)

    # Grid lines
    for i in range(N + 1):
        pygame.draw.line(screen, (220, 220, 220), (i * CELL_SIZE, 0), (i * CELL_SIZE, HEIGHT))
        pygame.draw.line(screen, (220, 220, 220), (0, i * CELL_SIZE), (WIDTH, i * CELL_SIZE))

        # ... inside your drawing loop (after drawing the grid and arrows) ...
    pygame.display.flip()

def initialize():
    global states, grid, policy, actions, returns, Q

    states = [(x, y, vx, vy)
              for y in range(N) for x in range(N) for vx in range(6) for vy in range(6)
              if grid[y][x] != 1]

    returns = {(s, a): 0 for s in states for a in actions}
    Q = {(s, a): 0 for s in states for a in actions}

def get_action(state, eps):
    # Epsilon-greedy
    if random.random() < eps:
        return random.randint(0, len(actions) - 1)

    q_vals = [Q.get((state, a), -1000) for a in range(len(actions))]
    max_q = max(q_vals)
    return random.choice([i for i, v in enumerate(q_vals) if v == max_q])

def runEpisode(exploring_start):
    if exploring_start and random.random() < epsilon:
        # Exploring Starts: Start anywhere on the track to spread knowledge
        track_cells = [(x, y) for y in range(N) for x in range(N) if grid[y][x] == 0]
        sx, sy = random.choice(track_cells)
        svx, svy = random.randint(0, 4), random.randint(0, 4)
    else:
        # Standard start line
        starts = [(x, y) for y in range(N) for x in range(N) if grid[y][x] == 2]
        sx, sy = random.choice(starts)
        svx, svy = 0, 0

    state = (sx, sy, svx, svy)
    trajectory = []

    # Increase steps to ensure they can find the far end of the track
    for _ in range(1000):
        action_idx = get_action(state, epsilon)
        ax, ay = actions[action_idx]

        vx_new = max(0, min(state[2] + ax, 4))
        vy_new = max(0, min(state[3] + ay, 4))
        if vx_new == 0 and vy_new == 0:  # Cannot stay at 0,0
            vx_new, vy_new = (state[2], state[3]) if state[2] + state[3] > 0 else (1, 0)

        # Movement with 50% Sutton-Barto displacement
        dx, dy = vx_new, vy_new
        if random.random() < 0.5:
            # Randomly move one extra cell forward/right
            dx += 1

        nx, ny = state[0] + dx, state[1] - dy

        # Collision/Finish Logic
        if nx < 0 or nx >= N or ny < 0 or ny >= N or grid[ny][nx] == 1:
            # Hit wall: Reset to start line per Sutton & Barto
            starts = [(x, y) for y in range(N) for x in range(N) if grid[y][x] == 2]
            state = (*random.choice(starts), 0, 0)
            reward = -1
        elif grid[ny][nx] == 3:
            # Crossed Finish Line
            trajectory.append((state, action_idx, -1))
            return trajectory, True
        else:
            reward = -1
            trajectory.append((state, action_idx, reward))
            state = (nx, ny, vx_new, vy_new)

    return trajectory, False  # Timed out

def train(i):
    global history, success_count
    is_exploring = i < 300000

    trajectory, finished = runEpisode(is_exploring)

    # Calculate G (Return)
    G = 0
    visited = set()

    for s, a, r in reversed(trajectory):
        G = r + gamma * G  # In this task, G is just -steps
        if (s, a) not in visited:
            if (s, a) not in returns:
                returns[(s, a)] = 0
                Q[(s, a)] = -500.0  # Optimistic initial value

            returns[(s, a)] += 1
            Q[(s, a)] += (G - Q[(s, a)]) / returns[(s, a)]
            visited.add((s, a))

    history.append(G)
    if finished: success_count += 1

    # Progress Display every 5000 episodes
    if i % 5000 == 0:
        avg_ret = sum(history) / len(history)
        win_rate = (success_count / i) * 100
        known_states = len(Q)
        print(f"{i:<10} | {avg_ret:<12.2f} | {win_rate:<9.1f}% | {known_states:<10}")
        update_heatmap()
        master_draw()

count = 1
running = True
master_draw()
while running:
    # 2. Event Handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE and mode < 4:
                mode += 1

                if mode == 4:
                    initialize()
                    print("initialized")

    # Check mouse state outside the event loop
    mouse_buttons = pygame.mouse.get_pressed()

    if mouse_buttons[0] and mode < 4:  # 0 is Left Click, 1 is Middle, 2 is Right
        mouse_pos = pygame.mouse.get_pos()
        x = mouse_pos[0] // CELL_SIZE
        y = mouse_pos[1] // CELL_SIZE

        # Guard clause to stay within grid bounds
        if 0 <= x < N and 0 <= y < N:
            grid[y][x] = mode

    if mode == 4:
        train(count)
        count += 1
pygame.quit()