"""Reinforce module for training the CASH controller."""


class CASHReinforce(object):

    def __init__(self, controller):
        self.controller = controller

    def fit(self, t_env, n_episodes=2, n_iter=10):
        for i_episode in range(n_episodes):
            t_env.sample_data_env()
            t_state = t_env.sample()
            for i in range(n_iter):
                pass
