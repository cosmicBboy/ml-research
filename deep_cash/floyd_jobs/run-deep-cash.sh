floyd run --env pytorch-0.3 $@ \
    ". ./.env && \
    python experiments/run_deep_cash.py \
    --output_fp=/output \
    --n_layers=3 \
    --beta=0.9 \
    --n_episodes=1000 \
    --n_iter=10 \
    --per_framework_time_limit=600 \
    --per_framework_memory_limit=10000 \
    --logger=floyd \
    --fit_verbose 0"
