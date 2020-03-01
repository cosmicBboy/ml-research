#!/bin/bash

# Syncs the outputs from floydhub
# Usage:
# `./sync_floyd_output.sh <job_number>`

rm -rf floyd_outputs/$1
mkdir -p floyd_outputs/$1
python floyd_scripts/cli.py get-output \
    nielsbantilan/projects/deep-cash/$1/output && \
    mv output.tar floyd_outputs/$1
