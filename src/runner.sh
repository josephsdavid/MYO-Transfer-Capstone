#!/bin/bash
#SBATCH -J deep_learning
#SBATCH -e result/issues
#SBATCH -o result/results.txt
#SBATCH -p  v100x8 --gres=gpu:2 --mem=40G
#SBATCH -t 400
#SBATCH -n 32
#SBATCH --exclusive


python test.py
