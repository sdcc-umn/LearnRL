import numpy as np


class Agent:
    def __init__(self):
        self.dict = self.create_dictionary()

    def create_dictionary(self):
        temp_dict = {}
        for state in range(16):
            for action in range(4):
                temp_dict[(state, action)] = (1, 1)
        return temp_dict

    def choose_action(self, obs):
        best_action = -1
        best_reward = -1
        for action in range(4):
            loss, win = self.dict[(obs, action)]
            percent = win/(win + loss + 0.0)
            if percent > best_reward:
                best_reward = percent
                best_action = action
        return best_action

    def learn(self, episode, reward):
        for pair in episode:
            old_loss, old_win = self.dict[pair]
            if reward > 0:
                loss, win = old_loss, old_win + 1
            else:
                loss, win = old_loss + 1, old_win
            self.dict[pair] = (loss, win)

