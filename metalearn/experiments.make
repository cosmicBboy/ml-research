.PHONY: metalearning-experiment

CMD=./floyd_scripts/train-metalearn.sh
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

metalearn-a2c-all-tasks:
	@${CMD} ${CONF}/experiment_2020-01-31-21-11-49_metalearn_a2c_binary_multiclass_regression.yml metalearn_a2c_all_tasks

metalearn-a2c-n-episodes:
	@${CMD} ${CONF}/experiment_2020-03-03-21-02-38_metalearn_a2c_binary_n_episodes.yml metalearn_a2c_binary_tune_n_episodes metalearn_a2c_binary_n_episodes

metalearn-a2c-layer-size:
	@${CMD} ${CONF}/experiment_2020-03-03-21-36-48_metalearn_a2c_binary_layer_sizes.yml metalearn_a2c_binary_layer_sizes

metalearn-a2c-controller-depth:
	@${CMD} ${CONF}/experiment_2020-03-03-21-37-52_metalearn_a2c_binary_controller_depth.yml metalearn_a2c_binary_controller_depth

metalearn-a2c-small-envs-clf:
	@${CMD} ${CONF}/experiment_2020-03-29-20-04-24_metalearn_a2c_clf_open_ml_6_metatraining.yml small_task_env_clf

metalearn-a2c-small-task-env-reg:
	@${CMD} ${CONF}/experiment_2020-03-29-20-38-59_metalearn_a2c_reg_open_ml_6_metatraining.yml small_task_env_reg