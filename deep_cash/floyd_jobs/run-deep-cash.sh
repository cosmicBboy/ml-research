# experiment: error reward is negative instead of zero
floyd run --env pytorch-0.3 --cpu2 \
    --message 'baseline_negative_error_reward=-0.1' \
    ". ./.env && \
    python experiments/run_deep_cash.py \
    --output_fp=/output \
    --n_trials=3 \
    --n_layers=30 \
    --beta=0.9 \
    --with_baseline \
    --multi_baseline \
    --normalize_reward \
    --error_reward=-0.1 \
    --n_episodes=500 \
    --n_iter=10 \
    --per_framework_time_limit=600 \
    --per_framework_memory_limit=10000 \
    --logger=floyd_multiprocess \
    --fit_verbose=0"
