from setuptools import setup

with open('README.md') as f:
    long_description = f.read()


setup(
    name="metalearn-rl",
    version="0.0.2",
    description="MetaRL-based Estimator using Task-encodings for AutoML",
    long_description=long_description,
    url="https://github.com/cosmicBboy/ml-research/tree/master/metalearn",
    packages=[
        "metalearn",
        "metalearn.components",
        "metalearn.data_environments",
        "metalearn.inference"],
    install_requires=[
        "click==6.7",
        "dill",
        "floyd-cli",
        "kaggle",
        "matplotlib",
        "numpy",
        "openml==0.7.0",
        "pandas==0.23.4",
        "pynisher",
        "torch==0.4.1",
        "scikit-learn==0.19.2",
        "scipy",
        "yamlordereddictloader",
    ],
    scripts=["bin/metalearn"],
)
