.PHONY: metalearning-experiment

CMD=./floyd_jobs/train-metalearn.sh
CONF=config/floyd


metalearning-baseline:
	@${CMD} ${CONF}/experiment_2019-10-02-01-21-54_metalearn_binary.yml metalearn_binary && \
	${CMD} ${CONF}/experiment_2019-10-02-01-32-59_metalearn_multiclass.yml metalearn_multiclass && \
	${CMD} ${CONF}/experiment_2019-10-02-01-33-39_metalearn_regression.yml metalearn_regression && \
	${CMD} ${CONF}/experiment_2019-10-02-01-39-00_metalearn_binary_multiclass_regression.yml metalearn_binary_multiclass_regression
