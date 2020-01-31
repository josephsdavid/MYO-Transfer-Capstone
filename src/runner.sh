#!/usr/bin/env bash
#SBATCH -J deep_test
#SBATCH -e result/testerr-%j.txt
#SBATCH -o result/testout.txt
#SBATCH -p  v100x8 --gres=gpu:2 --mem=40G
#SBATCH -t 400
#SBATCH -n 32
#SBATCH --mail-user josephsd@smu.edu
#SBATCH --mail-type=ALL


python test.py
