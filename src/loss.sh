#!/usr/bin/env bash
# read in a file
file=$1
# make sure encoding is proper
export LC_NUMERIC="en_US.UTF-8"
load_file () {
	# grep 'val_acc|DELIMETER THAT DENOTES A NEW MODEL'
	# CHANGE AS NEEDED
	# FOR US FOR EVERY NEW MODEL I PRINT "beginning of model xyz"
	x="$(grep 'val_accuracy|beginning' "$file" -P 2> /dev/null)"
	# tf prints a bunch of special characters, remove those
	out="$(sed  's/[^[:print:]\t]//g'<<< "$x" 2> /dev/null)"
	# return the cleaned result
	echo "$out"
}

# save cleaned result to variable
x=$(load_file)
# counter
outpath=0

# read in the cleaned result variable as LOSS (The cleaned result is backpiped in at the bottom, because bash = dumb)
while read LOSS; do
	if [[ $LOSS == *"begin"* ]]; then
		#outpath=$((outpath+1))
		# use this for getting a better model output
		outpath=$(awk '{print $(NF)}' <<< "$LOSS")
		# if you have print("beginning UNIQUE_MODEL_NAME")
		# for every fit statement in your python script, this will nicely name the
		# csvs
		# this prunts the csv header
		printf "%s,%s,%s,%s\n" "val_acc" "val_loss"  "train_acc"  "train_loss" > history/model_"$outpath".csv
	else
		# this grabs the value for each thing
		val_acc="$(awk '{print $(NF)}' <<< "$LOSS" )"
		t_acc="$(awk '{print $(NF-6)}' <<< "$LOSS" )"
		val_loss="$(awk '{print $(NF-3)}' <<< "$LOSS" )"
		t_loss="$(awk '{print $(NF-9)}' <<< "$LOSS" )"
		# this prints the result for every model as a column separated by commas
		# ending with a tidy csv
		printf "%s,%s,%s,%s\n" "$val_acc" "$val_loss" "$t_acc" "$t_loss" >> history/model_"$outpath".csv
		#printf "%s,%s,%s,%s\n" "$val_acc" "$val_loss" "$t_acc" "$t_loss" >> history/model_"$outpath".csv
	fi
done <<< "$x"
