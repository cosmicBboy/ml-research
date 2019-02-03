deps:
	conda install pytorch==0.3.1 torchvision -y -c pytorch
	conda install -c conda-forge \
		click \
		dash \
		dash-core-components \
		dash-html-components \
		dill \
		pandas \
		numpy \
		matplotlib \
		ipython \
		jupyter \
		jupyterlab \
		plotly \
		seaborn \
		scikit-learn==0.19.2 \
		ipykernel \
		pytest \
		floydhub \
		floyd-cli
	pip install pynisher openml==0.7.0
	pip install yamlordereddictloader
	pip install kaggle
	python -m ipykernel install --name 'ml-research' --display-name 'ml-research'
