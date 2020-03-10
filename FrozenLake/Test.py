import gym
import numpy as np
import FrozenLake.Test_Agent as Test_Agent
import matplotlib.pyplot as plt

from gym.envs.registration import register
register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=2.0, # optimum = .8196
)

MAX_STEPS = 25
MAX_EPISODES = 1000

env = gym.make('FrozenLakeNotSlippery-v0')
agent = Test_Agent.Agent()
rewards = []

for _ in range(MAX_EPISODES):
    episode = []
    obs = env.reset()
    step = 0
    done = False
    while not done:
        step += 1
        action = agent.choose_action(obs)
        episode.append((obs, action))
        obs, reward, done, info = env.step(action)
        if done or step > 25:
            rewards.append(reward)
            agent.learn(episode, reward)
            done = True

plt.plot(rewards)
plt.show()