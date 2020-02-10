#!/usr/bin/env bash
printf '\033[2J \033[H'
file=$1
counter=$2
while [ "$counter" -gt 0 ]
do
	printf '\033[2J \033[H'
	tail -n 42 "$file"
	sleep 1s
	counter=$((counter-1))
done
