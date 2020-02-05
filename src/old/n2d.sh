#!/usr/bin/env bash
#SBATCH -J n2d
#SBATCH -e result/n2d-%j.txt
#SBATCH -o result/n2d-%j.txt
#SBATCH -p  v100x8 --gres=gpu:1 --mem=80G
#SBATCH -t 10080
#SBATCH -n 32
#SBATCH --mail-user josephsd@smu.edu
#SBATCH --mail-type=ALL


python dimred.py

