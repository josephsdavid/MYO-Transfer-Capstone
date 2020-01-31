#!/usr/bin/env bash
printf '\033[2J'
counter=$1
while [ $counter -gt 0 ]
do
	printf '\033[H'
	squeue -u josephsd -l
	sleep 1s
	counter=$((counter-1))
done


