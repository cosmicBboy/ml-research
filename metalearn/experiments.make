.PHONY: metalearning-experiment

CMD=./floyd_jobs/train-metalearn.sh
CONF=config/floyd


metalearning-baseline: metalearn-binary metalearn-multiclass metalearn-regression metalearn-all-tasks

metalearn-binary:
	@${CMD} ${CONF}/experiment_2019-10-02-01-21-54_metalearn_binary.yml metalearn_binary

metalearn-multiclass:
	${CMD} ${CONF}/experiment_2019-10-02-01-32-59_metalearn_multiclass.yml metalearn_multiclass

metalearn-regression:
	${CMD} ${CONF}/experiment_2019-10-02-01-33-39_metalearn_regression.yml metalearn_regression

metalearn-all-tasks:
	${CMD} ${CONF}/experiment_2019-10-02-01-39-00_metalearn_binary_multiclass_regression.yml metalearn_binary_multiclass_regression

metalearn-binary-tuning:
	@${CMD} ${CONF}/experiment_2019-12-26-23-34-27_metalearn_binary_tuning.yml metalearn_binary_tuning

metalearn-a2c-binary:
	@${CMD} ${CONF}/experiment_2020-01-20-16-56-31_metalearn_a2c_binary.yml metalearn_a2c_binary
