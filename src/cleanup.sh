#!/usr/bin/env bash
file=$1
export LC_NUMERIC="en_US.UTF-8"
load_file () {
	x="$(grep 'val_accuracy|beginning' "$file" -P 2> /dev/null)"
	out="$(sed  's/[^[:print:]\t]//g'<<< "$x" 2> /dev/null)"
	echo "$out"
}

x=$(load_file)
count=0

while read LOSS; do
	if [[ $LOSS == *"begin"* ]]; then
		count=$((count+1))
		printf "%s,%s,%s,%s\n" "val_acc" "val_loss"  "train_acc"  "train_loss" > history/model_"$count".csv
	else
		for i in "${LOSS[@]}"; do
			val_acc="$(awk '{print $(NF)}' <<< "$i" )"
			t_acc="$(awk '{print $(NF-6)}' <<< "$i" )"
			val_loss="$(awk '{print $(NF-3)}' <<< "$i" )"
			t_loss="$(awk '{print $(NF-9)}' <<< "$i" )"

			printf "%d,%d,%d,%d\n" "$val_acc" "$val_loss" "$t_acc" "$t_loss" >> history/model_"$count".csv
			printf "%d,%d,%d,%d\n" "$val_acc" "$val_loss" "$t_acc" "$t_loss" >> history/model_"$count".csv
		done
	fi
done <<< "$x"
