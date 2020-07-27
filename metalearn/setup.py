from setuptools import setup

with open('README.md') as f:
    long_description = f.read()


setup(
    name="meta-ml",
    version="0.0.21",
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
        "colorlover",
        "click==7.0",
        "dill==0.3.1.1",
        "kaggle",
        "matplotlib",
        "numpy",
        "openml==0.9.0",
        "pandas==1.0.5",
        "plotly",
        "psutil",
        "torch==1.5.0",
        "scipy",
        "yamlordereddictloader",
    ],
    extras_require={
        "sklearn": [
            "scikit-learn>=0.21.2",
        ]
    },
    scripts=["bin/metalearn"],
)
