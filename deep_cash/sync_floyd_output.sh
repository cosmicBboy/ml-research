#!/bin/bash

# Syncs the outputs from floydhub
# Usage:
# `./sync_floyd_output.sh <job_number>`

rm -rf floyd_outputs/$1
mkdir -p floyd_outputs/$1
floyd data clone nielsbantilan/projects/deep-cash/$1/output
mv cash_controller_mlfs_trial_* floyd_outputs/$1
mv controller_trial_* floyd_outputs/$1
mv rnn_cash_controller_experiment.csv floyd_outputs/$1
mv fit_predict_error_logs.log floyd_outputs/$1
