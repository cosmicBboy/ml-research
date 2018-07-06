deps:
	conda install pytorch torchvision -y -c pytorch
	conda install -c conda-forge \
		dill \
		pandas \
		numpy \
		matplotlib \
		ipython \
		seaborn \
		scikit-learn \
		ipykernel \
		pytest \
		floydhub \
		floyd-cli
	pip install pynisher openml
	python -m ipykernel install --name 'ml-research' --display-name 'ml-research'
