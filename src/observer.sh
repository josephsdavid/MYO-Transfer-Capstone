#!/usr/bin/env bash
file=$1
counter=$2
while [ $counter -gt 0 ]
do
	printf '\033[H'
	cat $file
	sleep 1s
	counter=$((counter-1))
done
