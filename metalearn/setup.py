from setuptools import setup

with open('README.md') as f:
    long_description = f.read()


extras_require = {
    "sklearn": [
        "scikit-learn>=0.21.2",
    ],
    "tasks": [
        "openml==0.9.0",
        "kaggle",
        "psutil",
    ],
    "experiments": [
        "yamlordereddictloader",
    ]
}

extras_require["all"] = sorted(
    {v for req in extras_require.values() for v in req}
)

setup(
    name="meta-ml",
    version="0.0.24",
    description="MetaRL-based Estimator using Task-encodings for AutoML",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/cosmicBboy/ml-research/tree/master/metalearn",
    packages=[
        "metalearn",
        "metalearn.components",
        "metalearn.data_environments",
        "metalearn.data_environments.feature_maps",
        "metalearn.inference"],
    install_requires=[
        "click==7.0",
        "dill==0.3.1.1",
        "numpy",
        "pandas==1.0.5",
        "torch==1.5.0",
        "scipy",
    ],
    extras_require=extras_require,
    scripts=["bin/metalearn"],
)
