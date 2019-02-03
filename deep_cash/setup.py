from setuptools import setup


setup(
    name="deep-cash",
    version="0.0.1",
    description="Deep RL approach to Combined Algorithm Search and "
                "Hyperparameter Optimization",
    url="https://github.com/cosmicBboy/ml-research/tree/master/deep_cash",
    packages=[
        "deep_cash",
        "deep_cash.components",
        "deep_cash.data_environments",
        "deep_cash.inference"],
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
    scripts=["bin/deep-cash"],
)
