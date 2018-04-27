"""Example Usage of algorithm_space_structured module."""

from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from deep_cash import components
from deep_cash.algorithm_space import AlgorithmSpace

# create dataset to evaluate
X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=10,
    n_classes=2, shuffle=True)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# create an algorithm environment consisting of one data preprocessor,
# feature preprocessor, and classifier.
algorithm_space = AlgorithmSpace(
    data_preprocessors=[components.data_preprocessors.imputer()],
    feature_preprocessors=[components.feature_preprocessors.pca()],
    classifiers=[components.classifiers.logistic_regression()])
ml_frameworks = algorithm_space.framework_iterator()


# sample machine learning frameworks from the algorithm environment
i, j = 0, 0
num_samples = 100
num_candidates = 10
best_candidates = []
best_scores = []

while True:
    mlf = algorithm_space.sample_ml_framework()
    try:
        mlf.fit(X_train, y_train)
        train_score = roc_auc_score(y_train, mlf.predict(X_train))
        test_score = roc_auc_score(y_test, mlf.predict(X_test))
        # maintain the best candidates and their associated scores
        if len(best_candidates) < num_candidates:
            best_candidates.append(mlf)
            best_scores.append(test_score)
        else:
            min_index = best_scores.index(min(best_scores))
            if test_score > best_scores[min_index]:
                best_candidates[min_index] = mlf
                best_scores[min_index] = test_score
        print(
            "valid framework %d/%d > training auc: %0.02f - test auc: %0.02f" %
            (i, j, train_score, test_score))
        i += 1
        if i > num_samples:
            break
    except ValueError:
        # pass on frameworks that error out due to misconfiguration of
        # hyperparameters.
        pass
    j += 1


print("\nBest Models:")
print("------------")
for test_score, mlf in zip(best_scores, best_candidates):
    for step in mlf.steps:
        print(step)
    print("training auc: %0.02f - test auc: %0.02f\n" %
          (train_score, test_score))
