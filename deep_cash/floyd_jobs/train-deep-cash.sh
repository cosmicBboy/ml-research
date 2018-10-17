# this command trains a deep cash controller on floydhub
# TODO make --message and config file env vars
floyd run \
    --data nielsbantilan/datasets/kaggle-deep-cash-datasets:kaggle_data \
    ". ./.env && KAGGLE_CACHE_DIR=/floyd/input/kaggle_data " \
    "python ./bin/deep-cash run from-config $1" \
    --message "$2"
