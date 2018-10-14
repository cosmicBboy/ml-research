deps:
	conda install pytorch torchvision -y -c pytorch
	conda install -c conda-forge \
		click \
		dill \
		pandas \
		numpy \
		matplotlib \
		ipython \
		jupyter \
		jupyterlab \
		plotly \
		seaborn \
		scikit-learn \
		ipykernel \
		pytest \
		floydhub \
		floyd-cli
	pip install pynisher openml==0.7.0
	pip install yamlordereddictloader
	pip install kaggle
	python -m ipykernel install --name 'ml-research' --display-name 'ml-research'
