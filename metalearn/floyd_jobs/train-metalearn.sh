# this command trains a deep cash controller on floydhub
floyd run \
    --data nielsbantilan/datasets/kaggle-deep-cash-datasets:kaggle_data \
    ". ./floyd.env && python ./bin/metalearn run from-config $1" \
    --message "$2"
