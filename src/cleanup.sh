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

			printf "%s,%s,%s,%s\n" "$val_acc" "$val_loss" "$t_acc" "$t_loss" >> history/model_"$count".csv
			printf "%s,%s,%s,%s\n" "$val_acc" "$val_loss" "$t_acc" "$t_loss" >> history/model_"$count".csv
		done
	fi
done <<< "$x"

#while IFS=';' read  -ra LOSS; do
#	count=$((count+1))
#	printf "%s,%s,%s,%s\n" "val_acc" "val_loss"  "train_acc"  "train_loss" > model_"$count".csv
#	for i in "${LOSS[@]}"; do
#		val_acc="$(awk '{print $(NF)}' <<< "$i" )"
#		echo "$val_acc"
#		t_acc="$(awk '{print $(NF) }' <<< "$i" )"
#		val_loss="$(awk '{print $(NF) }' <<< "$i" )"
#		t_loss="$(awk '{print $(NF)  }' <<< "$i" )"
#		printf "%s,%s,%s,%s\n" "$val_acc" "$val_loss" "$t_acc" "$t_loss" >> model_"$count".csv
#	done
#done <<< "$x"


#nmodels="${#val_acc[@]}"
#
#for ((i=0; i<"$nmodels"; i++)); do
#	printf "%s,%s,%s,%s\n" "val_acc" "val_loss"  "train_acc"  "train_loss" > model_"$i".csv
#	for ((k=0; k<"${#val_acc[i]}"; k++)); do
#		printf "%s,%s,%s,%s\n" "$val_acc[$i]" "$val_loss[$i]" "$t_acc[$i]" "$t_loss[$i]">>model_"$i".csv
#	done
#
#done
#
#val_acc="$(awk '{print $(NF)}' <<< grep 'val_acc' "$x")"
#val_loss="$(awk '{print $(NF-3)}' <<< grep 'val_acc' "$x")"
#t_acc="$(awk '{print $(NF-6)}' <<< grep 'val_acc' "$x")"
#t_loss="$(awk '{print $(NF-9)}' <<< grep 'val_acc' "$x")"
#
#echo "$val_acc" "$val_loss" "$t_acc" "$t_loss"
