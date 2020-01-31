#!/bin/bash
#SBATCH -J deep_learning
#SBATCH -e result/testerr.txt
#SBATCH -o result/testout.txt
#SBATCH -p  v100x8 --gres=gpu:2 --mem=40G
#SBATCH -t 400
#SBATCH -n 32
#SBATCH --mail-user josephsd@smu.edu
#SBATCH --mail-type=ALL


python test.py
