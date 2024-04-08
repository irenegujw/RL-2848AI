from check_performance import get_average
from model.dqn_agent import DQNAgent
from game_env import GameEnv
import time

train_episodes = 30000


def train():
    episodes = train_episodes
    agent = DQNAgent()
    env = GameEnv()
    banned_action = []
    logs = []

    state, info = env.reset()
    while True:
        # use random select fill up replay buffer first
        action = agent.act(state, random=True)

        next_state, reward, terminated, truncated, info = env.step(action)

        agent.store_experience(state, action, reward, next_state, terminated)

        state = next_state.copy()

        if agent.replay_buffer.full:
            break

        if terminated or truncated:
            state, info = env.reset()

    for i in range(episodes):
        state, info = env.reset()
        # env.render()

        while True:
            action = agent.act(state, banned_action=banned_action)

            next_state, reward, terminated, truncated, info = env.step(action)

            agent.store_experience(state, action, reward, next_state, terminated)

            state = next_state.copy()

            if info["is_illegal_move"]:
                banned_action.append(action)
            else:
                banned_action = []

            loss = agent.update()
            # env.render()

            if terminated or truncated:
                print(
                    f"Episode: {i+1}, Score: {env.display_score}, Move: {env.move}, "
                    f"Illegal Move: {env.illegal_move}, Loss: {loss.numpy() if loss is not None else 0}"
                )
                if i > 0 and i % 500 == 0:
                    get_average(logs, 500)

                logs.append((i+1, env.display_score, env.move, loss.numpy() if loss is not None else 0))
                banned_action = []
                break


train()
