#!/usr/bin/env bash


directory_name="../data/ninaPro/np"

if [[ -d "${directory_name}" ]]; then
    echo "Directory already exists"
else
    echo "Making target np directory..."
    mkdir "${directory_name}"
fi
echo "Generating npy files..."
python -c 'from utils import ninaUtils as nu; nu.build_nps("../data/ninaPro")'
# For 16-channel data
# python -c 'from utils import ninaUtils as nu; nu.build_nps("../data/ninaPro", prefix="emg16", full_emg=True)'
echo "El Fin!"
