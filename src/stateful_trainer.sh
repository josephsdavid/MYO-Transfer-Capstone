#!/bin/bash
#SBATCH -J great_work
#SBATCH -e result/berr-%j.txt
#SBATCH -o result/bout-%j.txt
#SBATCH -p  v100x8 --gres=gpu:1 --mem=90G
#SBATCH -t 1440
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --mail-user josephsd@smu.edu
#SBATCH --mail-type=ALL

python stateful3.py
