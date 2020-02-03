#!/usr/bin/env bash
#SBATCH -J goodlstm
#SBATCH -e result/tuned-%j.txt
#SBATCH -o result/tuned-%j.txt
#SBATCH -p  v100x8 --gres=gpu:4 --mem=40G
#SBATCH -t 10080
#SBATCH -n 32
#SBATCH --mail-user josephsd@smu.edu
#SBATCH --mail-type=ALL


python test.py
