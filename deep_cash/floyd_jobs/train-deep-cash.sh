# this command trains a deep cash controller on floydhub
# TODO make --message and config file env vars
floyd run \
    ". ./.env && python ./bin/deep-cash run from-config $1" \
    --message "$2"
