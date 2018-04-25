"""Example Usage of algorithm_env_structured module."""

from deep_cash import components
from deep_cash.algorithm_env_structured import AlgorithmEnvStructured


# create an algorithm environment consisting of one data preprocessor,
# feature preprocessor, and classifier.
algorithm_env = AlgorithmEnvStructured(
    data_preprocessors=[components.data_preprocessors.imputer()],
    feature_preprocessors=[components.feature_preprocessors.pca()],
    classifiers=[components.classifiers.logistic_regression()])
ml_frameworks = algorithm_env.framework_iterator()

i = 0
print("Counting number of unique estimator and hyperparameter combinations.")
for f in ml_frameworks:
    i += 1
print("Number of ML frameworks in state space: %d" % i)
