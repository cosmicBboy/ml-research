floyd run --env pytorch-0.3 $@ \
    ". ./.env && \
    DEEP_CASH_OUT_PATH=/output \
    DEEP_CASH_N_EPISODES=150 \
    DEEP_CASH_N_ITER=100 \
    DEEP_CASH_LEARNING_RATE=0.0025 \
    DEEP_CASH_ERROR_REWARD=-1 \
    DEEP_CASH_LOGGER=floyd \
    DEEP_CASH_FIT_VERBOSE=0 \
    python examples/example_rnn_cash_controller.py"
