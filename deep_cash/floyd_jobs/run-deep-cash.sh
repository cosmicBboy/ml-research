floyd run --env pytorch-0.3 $@ \
    ". ./.env && \
    python experiments/run_deep_cash.py \
    --output_fp=/output \
    --n_trials=5 \
    --n_layers=30 \
    --beta=0.9 \
    --multi_baseline \
    --n_episodes=1000 \
    --n_iter=10 \
    --per_framework_time_limit=600 \
    --per_framework_memory_limit=10000 \
    --logger=floyd_multiprocess \
    --fit_verbose=0"
