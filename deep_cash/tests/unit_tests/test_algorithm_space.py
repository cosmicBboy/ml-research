from sklearn.pipeline import Pipeline

from metalearn.algorithm_space import AlgorithmSpace, \
    CLASSIFIER_MLF_SIGNATURE, REGRESSOR_MLF_SIGNATURE


MLF_SIGNATURES = [CLASSIFIER_MLF_SIGNATURE, REGRESSOR_MLF_SIGNATURE]


def _algorithm_space(random_state):
    return AlgorithmSpace(
        classifiers=None,
        regressors=None,
        hyperparam_with_none_token=False,
        random_state=random_state)


def _sample_mlfs(signature, n, random_state):
    mlfs = set()
    hyperparameters = set()
    algorithm_space = _algorithm_space(random_state)
    for i in range(n):
        mlf = algorithm_space.sample_ml_framework(signature)
        component_names, _ = zip(*mlf.steps)
        mlfs.add(component_names)
        hyperparameters.add(str(mlf.get_params()))
        assert isinstance(mlf, Pipeline)
    return mlfs, hyperparameters


def test_sample_ml_framework_random_state():
    """Test that random state yields repeatable samples."""
    for signature in MLF_SIGNATURES:
        # sampling the algorithm space with the same random state yields
        # the same mlfs
        mlfs, hyperparameters = _sample_mlfs(signature, 100, 9000)
        mlfs2, hyperparameters2 = _sample_mlfs(signature, 100, 9000)
        mlfs3, hyperparameters3 = _sample_mlfs(signature, 100, 25)
        assert mlfs == mlfs2
        assert hyperparameters == hyperparameters2
        assert mlfs != mlfs3
        assert hyperparameters != hyperparameters3


def test_sample_ml_framework_diversity():
    """Test mlf and hyperparameter diversity in samples."""
    for signature in MLF_SIGNATURES:
        mlf_diversity = []
        hyperparam_diversity = []
        for random_state in [51, 11, 1, 90, 265, 1523, 1233, 1111, 2222, 2]:
            mlfs, hyperparameters = _sample_mlfs(signature, 100, random_state)
            mlf_diversity.append(len(mlfs) / 100.0)
            hyperparam_diversity.append(len(hyperparameters) / 100.0)
        # for relatively small samples of mlfs, hyperparameter diversity
        # should be 100% because the search space is so large.
        assert sum(hyperparam_diversity) / len(hyperparam_diversity) == 1.0

        # mlf diversity should be >70%. Since the search space is smaller,
        # a sample of 100 can result in identical mlf components.
        assert sum(mlf_diversity) / len(mlf_diversity) >= 0.7
