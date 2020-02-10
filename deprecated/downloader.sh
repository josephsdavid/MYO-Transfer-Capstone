#!/usr/bin/env bash

f1=https://zenodo.org/record/1000116/files/s1.zip?download=1
f2=https://zenodo.org/record/1000116/files/s2.zip?download=1
f3=https://zenodo.org/record/1000116/files/s3.zip?download=1
f4=https://zenodo.org/record/1000116/files/s4.zip?download=1
f5=https://zenodo.org/record/1000116/files/s5.zip?download=1
f6=https://zenodo.org/record/1000116/files/s6.zip?download=1
f7=https://zenodo.org/record/1000116/files/s7.zip?download=1
f8=https://zenodo.org/record/1000116/files/s8.zip?download=1
f9=https://zenodo.org/record/1000116/files/s9.zip?download=1
f10=https://zenodo.org/record/1000116/files/s10.zip?download=1
g1=s1.zip
g2=s2.zip
g3=s3.zip
g4=s4.zip
g5=s5.zip
g6=s6.zip
g7=s7.zip
g8=s8.zip
g9=s9.zip
g10=s10.zip
wget  "${f1}" -O "${g1}"
wget  "${f2}" -O "${g2}"
wget  "${f3}" -O "${g3}"
wget  "${f4}" -O "${g4}"
wget  "${f5}" -O "${g5}"
wget  "${f6}" -O "${g6}"
wget  "${f7}" -O "${g7}"
wget  "${f8}" -O "${g8}"
wget  "${f9}" -O "${g9}"
wget  "${f10}" -O "${g10}"

mkdir nina
mv s*.zip nina
cd nina
for i in {1..10}
do
	unzip "s${i}.zip"
done
rm *.zip
