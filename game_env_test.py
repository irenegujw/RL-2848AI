import pygame
from main import GameEnv


def test_game_env():
    env = GameEnv(render_mode="human")

    # 测试重置游戏
    obs, info = env.reset()
    print(f"Initial observation: {obs}")
    print(f"Initial info: {info}")

    # 游戏循环
    while True:
        # 渲染游戏窗口
        env.render()

        # 处理事件
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

                # 测试执行动作
                obs, reward, done, truncated, info = env.step(action)
                print(f"Observation: {obs}")
                print(f"Reward: {reward}")
                print(f"Done: {done}")
                print(f"Truncated: {truncated}")
                print(f"Info: {info}")

                if done:
                    print("Game Over. Resetting the game.")
                    obs, info = env.reset()
                    print(f"Initial observation after reset: {obs}")
                    print(f"Initial info after reset: {info}")


if __name__ == "__main__":
    test_game_env()
