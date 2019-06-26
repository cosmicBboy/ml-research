from setuptools import setup

with open('README.md') as f:
    long_description = f.read()


setup(
    name="meta-ml",
    version="0.0.4",
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
        "click==6.7",
        "dill",
        "floyd-cli",
        "kaggle",
        "matplotlib",
        "numpy",
        "openml==0.9.0",
        "pandas==0.24.2",
        "pynisher",
        "torch==1.1.0.post2",
        "scikit-learn==0.21.2",
        "scipy",
        "yamlordereddictloader",
    ],
    scripts=["bin/metalearn"],
)
