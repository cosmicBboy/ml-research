# experiment: error reward is negative, medium neural network
# tuning the learning rate 0.0003
floyd run --env pytorch-0.3 --cpu2 \
    --message 'medium_nn0_learning_rate_0.0003' \
    ". ./.env && \
    python experiments/run_deep_cash.py \
    --output_fp=/output \
    --n_trials=2 \
    --input_size=60 \
    --hidden_size=60 \
    --output_size=60 \
    --n_layers=6 \
    --dropout_rate=0.2 \
    --beta=0.9 \
    --with_baseline \
    --multi_baseline \
    --normalize_reward \
    --n_episodes=500 \
    --n_iter=10 \
    --learning_rate=0.0003 \
    --error_reward=-0.1 \
    --per_framework_time_limit=600 \
    --per_framework_memory_limit=10000 \
    --logger=floyd_multiprocess \
    --fit_verbose=0"
