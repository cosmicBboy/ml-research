"""Evaluate a trained controller model."""

from collections import namedtuple
from sklearn.exceptions import NotFittedError
from torch.autograd import Variable

from .. import utils
from ..data_types import DataSourceType


TestSetResult = namedtuple(
    "TestSetResult", ["ml_framework", "reward", "test_score"])
InferenceResult = namedtuple(
    "InferenceResult",
    ["mlf_description", "reward", "action", "validation_score", "is_valid"])


class CASHInference(object):

    def __init__(self, controller, task_env):
        self.controller = utils.freeze_model(controller)
        self.t_env = task_env
        self._validation_scores = []
        self._successful_mlfs = []

    def evaluate_test_sets(self, dataset, mlfs):
        """Evaluate controller based on training dataenv test performance.

        Evaluation is based on the best mlfs that were found during training
        per data env.

        Note that for Kaggle data environments, the evaluation process is a
        little bit more involved since predictions need to be sent up to the
        kaggle platform to be evaluated

        TODO: should the controller be able to propose subsequent MLFs
        based on the reward of previous MLFs? Seems like this is introducing
        bias, but not sure exactly how.

        :param dict[str -> list] mlfs: key is dataset name and value is a list
            of sklearn.Pipeline objects.
        """
        inference_results = []
        data_env = [d for d in self.t_env.data_distribution
                    if d.name == dataset]
        if len(data_env) == 0:
            raise ValueError(
                "%s not in the task environment data distribution." %
                dataset)
        data_env = data_env[0]
        self.t_env.set_data_env(data_env)
        for mlf in mlfs:
            if data_env.source is DataSourceType.KAGGLE:
                if self.t_env.current_data_env.y_test is None:
                    # TODO: send a submission and get score via kaggle api.
                    print("%s is a kaggle data environment" % data_env.name)
                    reward, test_score = None, None
                else:
                    print(
                        "%s is a kaggle data environment using a holdout "
                        "partition training data to product test data. Note "
                        "this mode is for 'development', where the kaggle "
                        "test labels should only be accessible through the "
                        "kaggle API. For 'production' mode, specify a task "
                        "environment where test_set_config for KAGGLE data "
                        "source doesn't specify `test_size` or `random_state`"
                        % dataset)
            mlf, reward, test_score = self.t_env.score(
                mlf,
                self.t_env.current_data_env.X_test,
                self.t_env.current_data_env.y_test)
            if (mlf, reward, test_score) == (None, None, None):
                raise NotFittedError(
                    "%s is not a fitted MLF. Make sure that the experiment "
                    "serializes the fitted mlf pipeline." % mlf)
            inference_results.append(TestSetResult(mlf, reward, test_score))

        return inference_results

    def evaluate_training_data_envs(self, n=5, datasets=None, verbose=False):
        train_data_env_results = {}
        for train_data_env in self.t_env.data_distribution:
            if datasets and train_data_env.name not in datasets:
                continue
            self.t_env.set_data_env(train_data_env)
            train_data_env_results[train_data_env.name] = self.infer(
                n, verbose)
        return train_data_env_results

    def evaluate_test_data_envs(self, n=5, datasets=None, verbose=False):
        """Test data env evaluation.

        This workflow evaluates the controller with respect to a new dataset
        that it hasn't seen before. The new dataset has three partitions:
        training, validation, and test set. The controller should then propose
        MLFs, measured against the validation set. Since the controller is
        hypothesized to implement a meta-RL algorithm, it should be able to
        learn even if all the controller's weights are frozen.
        """
        test_data_env_results = {}
        for test_data_env in self.t_env.test_data_distribution:
            if datasets and test_data_env.name not in datasets:
                continue
            print("evaluating test data env: %s" % test_data_env.name)
            self.t_env.set_data_env(test_data_env)
            test_data_env_results[test_data_env.name] = self.infer(
                n, verbose)
        return test_data_env_results

    def infer(self, n, verbose):
        """Make inferences by sampling the task environment.

        :param int n: number of times to sample task environment.
        :param bool verbose: prints metric logs during training.
        :returns list[InferenceResult]: a named tuple
        """
        prev_reward, prev_action = 0, self.controller.init_action()
        inference_results = []
        n_valid_mlf = 0
        for i in range(n):
            mlf, inference = self._infer_iter(
                self.t_env.sample_task_state(data_env_partition="test"),
                self.t_env.current_data_env.target_type,
                prev_reward,
                prev_action)
            # TODO: serialize the top k mlfs here
            inference_results.append(inference)
            prev_reward, prev_action = inference.reward, inference.action
            n_valid_mlf += int(inference.is_valid)
            if verbose:
                print(
                    "iter %d - n valid mlf: %d/%d%s" % (
                        i, n_valid_mlf, i + 1, " " * 10),
                    sep=" ", end="\r", flush=True)
        return inference_results

    def _infer_iter(
            self, metafeature_tensor, target_type, prev_reward, prev_action):
        mlf, action_activation = self.propose_mlf(
            metafeature_tensor, target_type, prev_reward, prev_action)
        mlf, reward, validation_score, is_valid = self.evaluate_mlf(mlf)

        return mlf, InferenceResult(
            str(mlf.named_steps) if mlf is not None else mlf,
            reward, Variable(action_activation),
            validation_score, is_valid)

    def propose_mlf(
            self, task_state_tensor, target_type, prev_reward, prev_action):
        """Given a task state,  propose a machine learning framework."""
        actions, action_activation = self.controller.decode(
            init_input_tensor=prev_action,
            target_type=target_type,
            aux=utils.aux_tensor(prev_reward),
            metafeatures=Variable(task_state_tensor),
            init_hidden=self.controller.init_hidden())
        algorithms, hyperparameters = utils.get_mlf_components(actions)
        mlf = self.controller.a_space.create_ml_framework(
            algorithms, hyperparameters=hyperparameters,
            task_metadata=self.t_env.get_current_task_metadata())
        return mlf, action_activation.data

    def evaluate_mlf(self, mlf):
        """Evaluate actions on the validation set of the data environment."""
        mlf, reward, validation_score = self.t_env.evaluate(mlf)
        reward, is_valid = self.is_valid_mlf(reward)
        return mlf, reward, validation_score, is_valid

    def is_valid_mlf(self, reward):
        if reward is None:
            return self.t_env.error_reward, False
        return reward,  True
