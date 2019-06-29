deps:
	conda install pytorch==1.1.0 torchvision -y -c pytorch
	conda install -c conda-forge \
		click \
		dash \
		dash-core-components \
		dash-html-components \
		dill \
		pandas==0.24.2 \
		numpy \
		matplotlib \
		ipython \
		jupyter \
		jupyterlab \
		plotly \
		seaborn \
		scikit-learn==0.21.2 \
		ipykernel \
		pytest \
		floydhub \
		floyd-cli
	pip install pynisher openml==0.9.0
	pip install yamlordereddictloader
	pip install kaggle
	python -m ipykernel install --name 'ml-research' --display-name 'ml-research'
