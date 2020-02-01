#!/bin/bash
#SBATCH -J stateful
#SBATCH -e result/stateful_err.txt
#SBATCH -o result/stateful_out.txt
#SBATCH -p  v100x8 --gres=gpu:1 --mem=90G
#SBATCH -t 10080
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mail-user josephsd@smu.edu
#SBATCH --mail-type=ALL

python stateful.py
