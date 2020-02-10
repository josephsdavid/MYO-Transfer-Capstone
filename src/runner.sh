#!/usr/bin/env bash
#SBATCH -J 🌊wavenet
#SBATCH -e result/wavenet-%j.txt
#SBATCH -o result/wavenet-%j.txt
#SBATCH -p  v100x8 --gres=gpu:2 --mem=90G
#SBATCH -t 10080
#SBATCH -n 32
#SBATCH --mail-user josephsd@smu.edu
#SBATCH --mail-type=ALL

python waverunner.py
