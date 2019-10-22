.PHONY: unit-tests endtoend-tests tests

deps:
	pip install -e . && \
	pip install -r dev-requirements.txt

unit-tests:
	pytest tests/unit_tests

endtoend-tests:
	pytest tests/end_to_end_tests

tests: unit-tests endtoend-tests

upload-pypi-test:
	python setup.py sdist bdist_wheel && \
		twine upload --repository-url https://test.pypi.org/legacy/ dist/* && \
		rm -rf dist

upload-pypi:
	python setup.py sdist bdist_wheel && \
		twine upload dist/* && \
		rm -rf dist

clean-pyc:
	find . -name '*.pyc' -exec rm {} \;
