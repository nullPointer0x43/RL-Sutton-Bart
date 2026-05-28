import pygame
from pointCollector import pointCollector
from stable_baselines3 import DQN

# 1. Load your trained model
# Make sure the filename matches what you saved earlier
model = DQN.load("dqn_point_collector")

# 2. Initialize the Environment
env = pointCollector(size=20)  # Using a larger size for better visuals
obs, _ = env.reset()

# 3. Pygame Setup
pygame.init()
screen_size = 400
screen = pygame.display.set_mode((screen_size, screen_size))
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 18)

running = True
total_reward = 0

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # --- AI LOGIC ---
    # Tell the model to look at the current observation and pick an action
    # deterministic=True ensures the AI plays its "best" move without exploring
    action, _states = model.predict(obs, deterministic=True)

    # Apply the action to the environment
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    # --- DRAWING ---
    screen.fill((30, 30, 30))  # Dark background

    # Draw Target (Green Circle)
    tx, ty = env.target[0] * (screen_size / 20), env.target[1] * (screen_size / 20)
    pygame.draw.circle(screen, (0, 255, 0), (int(tx), int(ty)), 10)

    # Draw Agent (Blue Circle)
    ax, ay = env.agent[0] * (screen_size / 20), env.agent[1] * (screen_size / 20)
    pygame.draw.circle(screen, (0, 150, 255), (int(ax), int(ay)), 12)

    # Draw Velocity Vector (Small white line showing direction)
    vx, vy = env.agent[2] * 10, env.agent[3] * 10
    pygame.draw.line(screen, (255, 255, 255), (int(ax), int(ay)), (int(ax + vx), int(ay + vy)), 2)

    # Display Stats
    text = font.render(f"Reward: {total_reward:.2f}", True, (255, 255, 255))
    screen.blit(text, (10, 10))

    pygame.display.flip()

    # Check if episode ended
    if terminated or truncated:
        print(f"Episode Finished! Total Reward: {total_reward:.2f}")
        obs, _ = env.reset()
        total_reward = 0

    clock.tick(60)  # Run at 60 FPS

pygame.quit()