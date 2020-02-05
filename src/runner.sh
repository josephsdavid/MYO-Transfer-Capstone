#!/usr/bin/env bash
#SBATCH -J mtask
#SBATCH -e result/batched-%j.txt
#SBATCH -o result/batched-%j.txt
#SBATCH -p  v100x8 --gres=gpu:4 --mem=60G
#SBATCH -t 10080
#SBATCH -n 32
#SBATCH --mail-user josephsd@smu.edu
#SBATCH --mail-type=ALL

python genTest.py
