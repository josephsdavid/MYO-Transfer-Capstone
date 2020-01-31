#!/usr/bin/env bash
file=$1
counter=$2
while [ $counter -gt 0 ]
do
	printf '\033[2J'
	cat $file
	sleep 1s
	counter=$((counter-1))
done
