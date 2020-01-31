#!/bin/bash
#SBATCH -J lstm_search
#SBATCH -e result/tune_err.txt
#SBATCH -o result/tune_out.txt
#SBATCH -p  v100x8 --gres=gpu:2 --mem=60G
#SBATCH -t 10080
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mail-user josephsd@smu.edu
#SBATCH --mail-type=ALL

python tuner.py
