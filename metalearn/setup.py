from setuptools import setup

with open('README.md') as f:
    long_description = f.read()


# use fork of floyd-cli to support click==0.7
floyd_url = (
    "https://github.com/cosmicBboy/floyd-cli/"
    "tarball/master#egg=floyd-cli-0.11.17-fork"
)
floyd_cli = "floyd-cli @ %s" % floyd_url

setup(
    name="meta-ml",
    version="0.0.15",
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
        "cytoolz",
        "dash",
        "dash-core-components",
        "dash-html-components",
        "dill",
        floyd_cli,
        "kaggle",
        "matplotlib",
        "numpy",
        "openml==0.9.0",
        "pandas==0.24.2",
        "psutil",
        "pynisher",
        "torch==1.2.0",
        "scikit-learn==0.21.2",
        "scipy",
        "streamlit==0.47.4",
        "yamlordereddictloader",
    ],
    scripts=["bin/metalearn"],
)
