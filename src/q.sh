#!/usr/bin/env bash
counter=$1
while [ $counter -gt 0 ]
do
	printf '\033[2J'
	squeue -u josephsd -l
	sleep 1s
	counter=$((counter-1))
done


