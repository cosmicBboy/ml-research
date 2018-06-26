floyd run --env pytorch-0.3 $@ \
    ". ./.env && \
    DEEP_CASH_OUT_PATH=/output \
    DEEP_CASH_N_EPISODES=200 \
    DEEP_CASH_N_ITER=100 \
    DEEP_CASH_LEARNING_RATE=0.003 \
    DEEP_CASH_ERROR_REWARD=-1 \
    DEEP_CASH_LOGGER=floyd \
    DEEP_CASH_FIT_VERBOSE=0 \
    python experiments/tuning_exponential_mean_beta.py"
