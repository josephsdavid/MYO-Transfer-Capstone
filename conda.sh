#!/usr/bin/env bash

while read requirement; do conda install --yes -n capstone $requirement; done < requirements.txt
pip install EMD-signal
pip install keras-tuner
