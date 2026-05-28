# Reinforcement Learning: Sutton & Barto Implementations

This repository contains Python implementations of core algorithms, classic examples, and exercises from Richard S. Sutton and Andrew G. Barto's *Reinforcement Learning: An Introduction* (2nd Edition), culminating in a deep reinforcement learning baseline implementation.

This project covers the spectrum of RL development: from exact Dynamic Programming equations and tabular Tabular/Monte Carlo methods, to hybrid planning/learning architectures (Dyna) and Eligibility Traces, ending with modern Deep Q-Networks.

---

## Implementations

### 1. Multi-Armed Bandits (Chapter 2)
Focuses on the core exploration vs. exploitation trade-off under stationary and non-stationary settings.
* **$N$-Armed Bandit Evaluation:** Comparative evaluation framework tracking performance metrics over thousands of steps.
* **Gradient Bandit:** Implementation of preference-based learning using numerical preferences and softmax action selection rather than action-value estimates.
* **$\alpha$ vs. $\epsilon$ Variation Analysis:** Comprehensive benchmark tracking how tracking-step sizes ($\alpha$) react against static sample-average methods ($\epsilon$-greedy) when tracking non-stationary targets.

### 2. Dynamic Programming (Chapter 4)
Exact model-based solution methods assuming perfect knowledge of the environment's transition dynamics ($P$) and rewards ($R$).
* **Gridworld Evaluation:** Iterative Policy Evaluation for a simple grid-layout Markov Decision Process.
* **Jack's Car Rental (Example 4.2):** Policy Iteration managing non-linear dynamics, constraints, and Poisson distributions for car moving operations between two locations.
* **The Gambler’s Problem (Example 4.3):** Value iteration solving an undamped betting game where the policy mapping reveals striking non-linear, fractal threshold properties based on the win-probability $p_h$.

### 3. Monte Carlo Methods (Chapter 5)
First model-free learning methods, estimating value functions directly from sampling entire episodes of experience.
* **Blackjack Environment:** * **On-Policy MC:** Exploring starts and $\epsilon$-soft policy evaluation/improvement loops.
  * **Off-Policy MC via Importance Sampling:** Separation of target policy (greedy) from behavior policy (exploratory), tracking both *Ordinary* and *Weighted* importance sampling ratios to optimize variance vs. bias.
* **Racetrack (Exercise 5.12):** Off-policy Monte Carlo control tracking discrete positional states and 2D directional velocities $(v_x, v_y)$ to teach an agent to find the optimal racing line while handling zero-velocity terminal crashes.

### 4. Temporal-Difference Learning (Chapter 6)
Combining the model-free sampling of Monte Carlo with the boot-strapping updating style of Dynamic Programming ($TD(0)$).
* **Simple MDPs:** Minimal proof-of-concept setups isolating value convergence rates under basic Markov chain steps.
* **Tic-Tac-Toe TD:** Tabular TD-learning agent playing games against an opponent, using value updates to evaluate boards incrementally per turn.
* **Windy Gridworld (Example 6.5) + King's Moves:** * Implementation of On-policy **SARSA** controlling navigation across a coordinate grid with an upward crosswind force.
  * Extends the classic setup to support **King's Moves** (8-directional movement variants) to evaluate pathing speedups.

### 5. Planning and Learning with Tabular Methods (Chapter 8)
Architectures that integrate model-free model updates with a simulated model framework to perform background planning.
* **Dyna-Q Maze:** Classic Dyna-Q execution running real environment steps alongside $N$ simulated background planning steps per real action to dramatically accelerate value convergence.
* **Dyna-Q+:** Implementation of environment exploration incentives. Adds an exploration bonus reward ($r + \kappa \sqrt{\tau}$) based on time steps elapsed ($\tau$) since a state-action pair was last visited, showing how the agent reacts dynamically when shortcuts open or close in a changing maze layout.

### 6. Eligibility Traces (Chapter 12)
Bridging the mechanistic gap between $TD(0)$ bootstrapping and full-sequence Monte Carlo returns using backward-view tracing vectors.
* **Random Walk Framework:** Evaluating the unifying properties of $\lambda$-returns over a bounded horizontal chain.
* **Backward-View $TD(\lambda)$:** Utilizing accumulating or replacing eligibility traces to distribute credit backward along a state-action trajectory.
* **Windy Gridworld via SARSA($\lambda$):** Performance comparison tracking how efficiently eligibility steps propagate reward backward down path trajectories compared to one-step standard SARSA.

### 7. Deep Reinforcement Learning
Transitioning from tabular tracking state spaces into high-dimensional neural network function approximations.
* **Deep Q-Network (DQN):** A robust baseline script implementing deep Q-learning via `stable-baselines3`. Leverages deep neural architectures, experience replay buffers, and target networks to stabilize value approximations in challenging control spaces.

---

## Installation & Setup

1. Clone this repository:
   ```bash
   git clone [https://github.com/nullPointer0x43/RL-Sutton-Barto.git](https://github.com/nullPointer0x43/RL-Sutton-Barto.git)
   cd RL-Sutton-Barto