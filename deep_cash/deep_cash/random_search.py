"""Modules implementing random search."""

from .algorithm_space import TARGET_TYPE_TO_MLF_SIGNATURE


class CASHRandomSearch(object):

    def __init__(self, a_space, t_env):
        """Initialize random search ML framework fitter."""
        self.a_space = a_space
        self.t_env = t_env

    def fit(self, n_episodes: int, n_iter: int):
        # initialize performance metrics data structures
        for i_episode in range(n_episodes):
            # TODO: not sure if this for loop should be parallelized
            self.t_env.sample_data_env()
            mlf_signature = TARGET_TYPE_TO_MLF_SIGNATURE[
                self.t_env.current_data_env.target_type]
            mlfs, rewards, scores = [], [], []
            for i_iter in range(n_iter):
                # TODO: this for loop can be parallelized
                self.t_env.sample_task_state()
                mlf = self.a_space.sample_ml_framework(mlf_signature)
                mlf, reward, score = self.t_env.evaluate(mlf)
                if mlf is not None:
                    mlfs.append(mlf)
                    rewards.append(reward)
                    scores.append(score)

        pass

    def history(self):
        pass
