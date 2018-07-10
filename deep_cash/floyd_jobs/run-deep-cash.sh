floyd run --env pytorch-0.3 $@ \
    ". ./.env && \
    python experiments/run_deep_cash.py \
    --output_fp=/output \
    --n_layers=3 \
    --beta=0.99 \
    --n_episodes=2000 \
    --per_framework_time_limit=180 \
    --per_framework_memory_limit=3077 \
    --logger=floyd \
    --fit_verbose 0"
