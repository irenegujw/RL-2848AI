import pygame
from game_env import GameEnv


def test_game_env():
    env = GameEnv(render_mode="human")

    # initial game environment
    obs, info = env.reset()
    print(f"Initial observation: {obs}")
    print(f"Initial info: {info}")

    while True:
        # render game
        env.render()

        # processing keyboard event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action = 0
                elif event.key == pygame.K_DOWN:
                    action = 1
                elif event.key == pygame.K_LEFT:
                    action = 2
                elif event.key == pygame.K_RIGHT:
                    action = 3
                else:
                    continue

                # executing action in game environment
                obs, reward, done, truncated, info = env.step(action)
                print(f"Observation: {obs}")
                print(f"Reward: {reward}")
                print(f"Done: {done}")
                print(f"Truncated: {truncated}")
                print(f"Info: {info}")
                env.render()

                if done:
                    print("Game Over. Resetting the game.")
                    obs, info = env.reset()
                    print(f"Initial observation after reset: {obs}")
                    print(f"Initial info after reset: {info}")


if __name__ == "__main__":
    test_game_env()
