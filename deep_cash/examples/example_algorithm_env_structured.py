"""Example Usage of algorithm_env_structured module."""

from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from deep_cash import components
from deep_cash.algorithm_env_structured import AlgorithmEnvStructured

# create dataset to evaluate
X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=10,
    n_classes=2, shuffle=True)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# create an algorithm environment consisting of one data preprocessor,
# feature preprocessor, and classifier.
algorithm_env = AlgorithmEnvStructured(
    data_preprocessors=[components.data_preprocessors.imputer()],
    feature_preprocessors=[components.feature_preprocessors.pca()],
    classifiers=[components.classifiers.logistic_regression()])
ml_frameworks = algorithm_env.framework_iterator()


# sample machine learning frameworks from the algorithm environment
i, j = 0, 0
num_samples = 100
num_candidates = 10
candidates = []

while True:
    mlf = algorithm_env.sample_ml_framework()
    try:
        mlf.fit(X_train, y_train)
        train_score = roc_auc_score(y_train, mlf.predict(X_train))
        test_score = roc_auc_score(y_test, mlf.predict(X_test))
        if len(candidates) == 0:
            candidates.append((mlf, train_score, test_score))
        else:
            # insert best-performing candidates into stack
            for index, c in enumerate(candidates):
                if test_score > c[2]:
                    candidates.insert(index, (mlf, train_score, test_score))
                    break
                elif len(candidates) <= num_candidates:
                    candidates.append((mlf, train_score, test_score))
                    break
                else:
                    break
        print(
            "valid framework %d/%d > training auc: %0.02f - test auc: %0.02f" %
            (i, j, train_score, test_score))
        i += 1
        if i > num_samples:
            break
    except ValueError:
        pass
    j += 1


print("\nBest Models:")
print("------------")
for mlf, train_score, test_score in candidates:
    for step in mlf.steps:
        print(step)
    print("training auc: %0.02f - test auc: %0.02f\n" %
          (train_score, test_score))
