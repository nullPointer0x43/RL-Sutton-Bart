import numpy as np
import gymnasium as gym
from typing import Optional
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
import os

class pointCollector(gym.Env):
    def __init__(self, size = 20):
        self.size = size

        self.agent = np.array([-1, -1, -1, -1], dtype = np.float32) # x, y, vx, vy
        self.target = np.array([-1, -1], dtype = np.float32) # x, y
        self.maxVel = 1

        low = np.array([0, 0, -self.maxVel, -self.maxVel, 0, 0], dtype=np.float32)
        high = np.array([size, size, self.maxVel, self.maxVel, size, size], dtype=np.float32)

        self.observation_space = gym.spaces.Box(low=low, high=high, shape=(6,), dtype=np.float32)

        self.action_space = gym.spaces.Discrete(4)

        self.actionToDirection = {
            0: np.array([0, 1]),   # Move right (column + 1)
            1: np.array([-1, 0]),  # Move up (row - 1)
            2: np.array([0, -1]),  # Move left (column - 1)
            3: np.array([1, 0]),   # Move down (row + 1)
        }

        self.current_step = 0
        self.max_steps = 500

        self.decay = 0.8

    def _get_obs(self):
        return np.concat([self.agent, self.target], axis = 0)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        x = self.np_random.uniform(0, self.size)
        y = self.np_random.uniform(0, self.size)
        vx = 0
        vy = 0
        self.agent = np.array([x, y, vx, vy], dtype = np.float32)

        target_x = x
        target_y = y
        while target_x == x or target_y == y:
            target_x = self.np_random.uniform(0, self.size)
            target_y = self.np_random.uniform(0, self.size)
        self.target = np.array([target_x, target_y], dtype = np.float32)

        self.current_step = 0

        observation = self._get_obs()

        return observation, {}

    def step(self, action):
        actionVec = self.actionToDirection[int(action)] * self.maxVel

        self.agent[2] = np.clip(self.agent[2] + actionVec[0], -self.maxVel, self.maxVel)
        self.agent[3] = np.clip(self.agent[3] + actionVec[1], -self.maxVel, self.maxVel)

        self.agent[2] *= self.decay
        self.agent[3] *= self.decay

        self.agent[0] = np.clip(self.agent[0] + self.agent[2], 0, self.size)
        self.agent[1] = np.clip(self.agent[1] + self.agent[3], 0, self.size)

        distance = np.linalg.norm(self.agent[:2] - self.target)

        # Small negative reward every step to encourage speed
        reward = -0.1

        # 7. Check if Goal Reached (Terminated)
        terminated = False
        if distance < 1:  # Threshold for "collecting" the point
            reward += 100.0
            terminated = True

        self.current_step += 1
        truncated = self.current_step >= self.max_steps

        observation = self._get_obs()

        return observation, reward, terminated, truncated, {}

if __name__ == '__main__':
    env = pointCollector(size=20)
    log_dir = "./logs/"
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, log_dir)

    #Best Parameters: {'lr': 0.00011632772479269217, 'gamma': 0.95, 'exploration_final_eps': 0.05057550920238869, 'buffer_size': 5000, 'batch_size': 64}
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=5e-5,  # Lowered for stability
        buffer_size=50000,  # Larger to keep "good" memories longer
        learning_starts=2000,
        batch_size=128,  # Larger batches provide more stable gradients
        gamma=0.99,  # High gamma for long-term planning
        target_update_interval=10000,
        exploration_fraction=0.5,  # Longer exploration phase
        verbose=1
    )

    # 3. Train the agent
    print("Training started...")
    model.learn(total_timesteps=100000)

    # 4. Save the model
    model.save("dqn_point_collector")
    print("Model saved!")