#!/bin/bash

# Syncs the outputs from floydhub
# Usage:
# `./sync_floyd_output.sh <job_number>`

rm -rf floyd_outputs/$1
mkdir -p floyd_outputs/$1
floyd data clone nielsbantilan/projects/deep-cash/$1/output
mv metalearn_controller_mlfs_trial_* floyd_outputs/$1
mv rnn_cash_*.csv floyd_outputs/$1
mv fit_predict_error_*.log floyd_outputs/$1

if ls controller_trial_* 1> /dev/null 2>&1; then
    # only present for cash controller experiments
    mv controller_trial_* floyd_outputs/$1
fi