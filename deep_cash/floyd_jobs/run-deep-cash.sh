# experiment: error reward is negative, small neural network
# set entropy coef to 0.8
floyd run --env pytorch-0.3 --cpu2 \
    --message 'small_nn0_entropy_coef_0.8' \
    ". ./.env && \
    python experiments/run_deep_cash.py \
    --output_fp=/output \
    --n_trials=1 \
    --input_size=30 \
    --hidden_size=30 \
    --output_size=30 \
    --n_layers=3 \
    --dropout_rate=0.3 \
    --beta=0.7 \
    --entropy_coef=0.2 \
    --with_baseline \
    --multi_baseline \
    --normalize_reward \
    --n_episodes=500 \
    --n_iter=16 \
    --learning_rate=0.003 \
    --error_reward=-0.05 \
    --per_framework_time_limit=180 \
    --per_framework_memory_limit=5000 \
    --logger=floyd \
    --fit_verbose=0"
