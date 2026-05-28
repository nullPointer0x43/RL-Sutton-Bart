import random
import collections

# --- Configuration ---
N = 15
GAMMA = 1.0
ITERATIONS = 500000
MAX_STEPS = 1000
epsilon = 0.2

# Stats for progress tracking
history = collections.deque(maxlen=1000)
success_count = 0

# Actions: increments to velocity (-1, 0, 1)
action_space = [(ax, ay) for ax in [-1, 0, 1] for ay in [-1, 0, 1]]

# 0: track, 1: wall, 2: start, 3: finish
grid = [[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 3], [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3], [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3], [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3], [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1]]

Q = {}
returns_n = {}


def get_action(state, eps):
    # Epsilon-greedy
    if random.random() < eps:
        return random.randint(0, len(action_space) - 1)

    q_vals = [Q.get((state, a), -1000) for a in range(len(action_space))]
    max_q = max(q_vals)
    return random.choice([i for i, v in enumerate(q_vals) if v == max_q])


def run_episode(exploring_start):
    # Determine Start State
    if exploring_start and random.random() < 0.5:
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

    for _ in range(MAX_STEPS):
        action_idx = get_action(state, epsilon)
        ax, ay = action_space[action_idx]

        # Physics update
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


# --- Training Loop with Progress Indicator ---
print(f"{'Iter':<10} | {'Avg Return':<12} | {'Win Rate':<10} | {'Knowledge':<10}")
print("-" * 55)

for i in range(1, ITERATIONS + 1):
    # Phase 1: Exploring Starts (First 60% of iterations)
    # Phase 2: Fine-tuning from Start Line (Final 40%)
    is_exploring = i < (ITERATIONS * 0.6)

    trajectory, finished = run_episode(is_exploring)

    # Calculate G (Return)
    G = 0
    visited = set()
    for s, a, r in reversed(trajectory):
        G = r + GAMMA * G  # In this task, G is just -steps
        if (s, a) not in visited:
            if (s, a) not in returns_n:
                returns_n[(s, a)] = 0
                Q[(s, a)] = -500.0  # Optimistic initial value

            returns_n[(s, a)] += 1
            Q[(s, a)] += (G - Q[(s, a)]) / returns_n[(s, a)]
            visited.add((s, a))

    history.append(G)
    if finished: success_count += 1

    # Progress Display every 5000 episodes
    if i % 5000 == 0:
        avg_ret = sum(history) / len(history)
        win_rate = (success_count / i) * 100
        known_states = len(Q)
        print(f"{i:<10} | {avg_ret:<12.2f} | {win_rate:<9.1f}% | {known_states:<10}")

print("\nTraining Complete.")