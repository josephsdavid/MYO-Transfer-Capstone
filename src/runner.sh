#!/bin/bash
#SBATCH -J deep_learning
#SBATCH -e result/gptesterror-%j.txt
#SBATCH -o result/gptestoutput-%j.txt
#SBATCH -p gpgpu-1 --gres=gpu:2 --mem=40G
#SBATCH -t 400
#SBATCH -n 32
#SBATCH --exclusive


python test.py
